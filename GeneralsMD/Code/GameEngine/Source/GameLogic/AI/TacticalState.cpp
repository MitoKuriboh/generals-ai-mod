/*
**	Command & Conquer Generals Zero Hour(tm)
**	Copyright 2025 Electronic Arts Inc.
**
**	This program is free software: you can redistribute it and/or modify
**	it under the terms of the GNU General Public License as published by
**	the Free Software Foundation, either version 3 of the License, or
**	(at your option) any later version.
**
**	This program is distributed in the hope that it will be useful,
**	but WITHOUT ANY WARRANTY; without even the implied warranty of
**	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**	GNU General Public License for more details.
**
**	You should have received a copy of the GNU General Public License
**	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  Tactical State - Implementation                                           //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

// TacticalState.cpp
// Team-level state for tactical decisions
// Author: Mito, 2025

#include "PreRTS.h"
#include "GameLogic/TacticalState.h"
#include "Common/Team.h"
#include "Common/Player.h"
#include "GameLogic/Object.h"
#include "GameLogic/GameLogic.h"
#include "GameLogic/PartitionManager.h"
#include "GameLogic/ObjectIter.h"
#include "GameLogic/ExperienceTracker.h"
#include "GameLogic/Module/BodyModule.h"

#include <cstring>
#include <cmath>

// =============================================================================
// TacticalState Implementation
// =============================================================================

void TacticalState::clear()
{
    memset(this, 0, sizeof(TacticalState));
}

void TacticalState::toFloatArray(Real* outArray) const
{
    // Copy all 64 floats in order
    Int idx = 0;

    // Strategy embedding (8)
    for (Int i = 0; i < 8; i++)
        outArray[idx++] = strategyEmbedding[i];

    // Team composition (12)
    for (Int i = 0; i < 3; i++) outArray[idx++] = teamInfantry[i];
    for (Int i = 0; i < 3; i++) outArray[idx++] = teamVehicles[i];
    for (Int i = 0; i < 3; i++) outArray[idx++] = teamAircraft[i];
    for (Int i = 0; i < 3; i++) outArray[idx++] = teamMixed[i];

    // Team status (8)
    outArray[idx++] = teamHealth;
    outArray[idx++] = ammunition;
    outArray[idx++] = cohesion;
    outArray[idx++] = experience;
    outArray[idx++] = distToObjective;
    outArray[idx++] = distToBase;
    outArray[idx++] = underFire;
    outArray[idx++] = hasTransport;

    // Situational (16)
    for (Int i = 0; i < 4; i++) outArray[idx++] = nearbyEnemies[i];
    for (Int i = 0; i < 4; i++) outArray[idx++] = nearbyAllies[i];
    outArray[idx++] = terrainAdvantage;
    outArray[idx++] = threatLevel;
    outArray[idx++] = targetValue;
    outArray[idx++] = supplyDist;
    outArray[idx++] = retreatPath;
    outArray[idx++] = reinforcePossible;
    outArray[idx++] = specialReady;
    outArray[idx++] = padding1;

    // Objective (8)
    outArray[idx++] = objectiveType;
    outArray[idx++] = objectiveX;
    outArray[idx++] = objectiveY;
    outArray[idx++] = priority;
    outArray[idx++] = progress;
    outArray[idx++] = timeOnObjective;
    outArray[idx++] = padding2;
    outArray[idx++] = padding3;

    // Temporal (4)
    outArray[idx++] = timeSinceEngagement;
    outArray[idx++] = timeSinceCommand;
    outArray[idx++] = framesSinceSpawn;
    outArray[idx++] = padding4;

    // Additional padding (8)
    outArray[idx++] = padding5;
    outArray[idx++] = padding6;
    outArray[idx++] = padding7;
    outArray[idx++] = padding8;
    outArray[idx++] = padding9;
    outArray[idx++] = padding10;
    outArray[idx++] = padding11;
    outArray[idx++] = padding12;
}

// =============================================================================
// TacticalCommand Implementation
// =============================================================================

void TacticalCommand::clear()
{
    action = TACTICAL_HOLD;
    targetX = 0.5f;
    targetY = 0.5f;
    attitude = 0.5f;
    valid = FALSE;
    teamId = -1;
}

// =============================================================================
// State Building Functions
// =============================================================================

// Helper struct for iterating team members
struct TeamMemberData
{
    Int infantryCount;
    Int vehicleCount;
    Int aircraftCount;
    Int otherCount;

    Real totalHealth;
    Real totalMaxHealth;
    Int unitCount;

    Real totalExperience;

    Coord3D centerPos;
    Real maxSpread;

    Bool hasTransport;
    Bool underFire;

    TeamMemberData()
        : infantryCount(0), vehicleCount(0), aircraftCount(0), otherCount(0)
        , totalHealth(0), totalMaxHealth(0), unitCount(0)
        , totalExperience(0), maxSpread(0), hasTransport(false), underFire(false)
    {
        centerPos.x = centerPos.y = centerPos.z = 0;
    }
};

static void collectTeamMemberData(Object* obj, void* userData)
{
    TeamMemberData* data = (TeamMemberData*)userData;
    if (!obj || obj->isEffectivelyDead())
        return;

    data->unitCount++;

    // Accumulate position for center calculation
    const Coord3D* pos = obj->getPosition();
    data->centerPos.x += pos->x;
    data->centerPos.y += pos->y;
    data->centerPos.z += pos->z;

    // Get health
    BodyModuleInterface* body = obj->getBodyModule();
    if (body)
    {
        data->totalHealth += body->getHealth();
        data->totalMaxHealth += body->getMaxHealth();
    }

    // Classify unit type
    if (obj->isKindOf(KINDOF_INFANTRY))
        data->infantryCount++;
    else if (obj->isKindOf(KINDOF_VEHICLE))
        data->vehicleCount++;
    else if (obj->isKindOf(KINDOF_AIRCRAFT))
        data->aircraftCount++;
    else
        data->otherCount++;

    // Check for transport
    if (obj->isKindOf(KINDOF_TRANSPORT))
        data->hasTransport = true;

    // Check if under fire
    if (obj->getBodyModule() && obj->getBodyModule()->getLastDamageTimestamp() > 0)
    {
        UnsignedInt currentFrame = TheGameLogic->getFrame();
        UnsignedInt lastDamage = obj->getBodyModule()->getLastDamageTimestamp();
        if (currentFrame - lastDamage < 60)  // Damaged in last 2 seconds
            data->underFire = true;
    }

    // Experience/veterancy
    ExperienceTracker* exp = obj->getExperienceTracker();
    if (exp)
    {
        data->totalExperience += (Real)exp->getVeterancyLevel() / 3.0f;  // Normalize to 0-1
    }
}

static void calculateSpread(Object* obj, void* userData)
{
    Real* params = (Real*)userData;
    Real centerX = params[0];
    Real centerY = params[1];
    Real* maxSpread = (Real*)&params[2];

    if (!obj || obj->isEffectivelyDead())
        return;

    const Coord3D* pos = obj->getPosition();
    Real dx = pos->x - centerX;
    Real dy = pos->y - centerY;
    Real dist = sqrtf(dx * dx + dy * dy);

    if (dist > *maxSpread)
        *maxSpread = dist;
}

void buildTacticalState(
    Team* team,
    const Real* strategicOutput,
    Player* player,
    TacticalState& outState)
{
    outState.clear();

    if (!team)
        return;

    // Copy strategic embedding
    if (strategicOutput)
    {
        for (Int i = 0; i < 8; i++)
            outState.strategyEmbedding[i] = strategicOutput[i];
    }

    // Collect team member data
    TeamMemberData data;
    team->iterateObjects(collectTeamMemberData, &data);

    if (data.unitCount == 0)
        return;

    // Calculate center position
    data.centerPos.x /= data.unitCount;
    data.centerPos.y /= data.unitCount;
    data.centerPos.z /= data.unitCount;

    // Calculate spread (cohesion)
    Real spreadParams[3] = { data.centerPos.x, data.centerPos.y, 0 };
    team->iterateObjects(calculateSpread, spreadParams);
    Real maxSpread = spreadParams[2];

    // Normalize counts (0-20 units expected)
    Real totalUnits = (Real)(data.infantryCount + data.vehicleCount + data.aircraftCount + data.otherCount);
    (void)totalUnits;  // Used for reference, individual counts normalized directly below

    // Team composition - [count normalized, health ~1, ready ~1]
    outState.teamInfantry[0] = fminf((Real)data.infantryCount / 10.0f, 1.0f);
    outState.teamInfantry[1] = data.totalMaxHealth > 0 ? data.totalHealth / data.totalMaxHealth : 1.0f;
    outState.teamInfantry[2] = 1.0f;  // Assume ready

    outState.teamVehicles[0] = fminf((Real)data.vehicleCount / 10.0f, 1.0f);
    outState.teamVehicles[1] = outState.teamInfantry[1];  // Same health ratio
    outState.teamVehicles[2] = 1.0f;

    outState.teamAircraft[0] = fminf((Real)data.aircraftCount / 5.0f, 1.0f);
    outState.teamAircraft[1] = outState.teamInfantry[1];
    outState.teamAircraft[2] = 1.0f;

    outState.teamMixed[0] = fminf((Real)data.otherCount / 5.0f, 1.0f);
    outState.teamMixed[1] = outState.teamInfantry[1];
    outState.teamMixed[2] = 1.0f;

    // Team status
    outState.teamHealth = data.totalMaxHealth > 0 ? data.totalHealth / data.totalMaxHealth : 1.0f;
    outState.ammunition = 1.0f;  // TODO: Track actual ammo
    outState.cohesion = 1.0f - fminf(maxSpread / 500.0f, 1.0f);  // 500 units = max spread
    outState.experience = data.unitCount > 0 ? data.totalExperience / data.unitCount : 0;

    // Distance to objective - use team's current waypoint or target
    const Coord3D* teamPos = team->getEstimateTeamPosition();
    if (teamPos)
    {
        // TODO: Get actual objective position
        outState.distToObjective = 0.5f;  // Default mid-range
    }

    // Distance to base
    if (player && teamPos)
    {
        // FIX: Get actual base position from player's command center
        Coord3D basePos = { 0, 0, 0 };
        Bool foundBase = false;

        for (Object* obj = TheGameLogic->getFirstObject(); obj; obj = obj->getNextObject())
        {
            if (obj->getControllingPlayer() != player) continue;
            if (obj->isKindOf(KINDOF_COMMANDCENTER) && !obj->isEffectivelyDead())
            {
                const Coord3D* pos = obj->getPosition();
                if (pos)
                {
                    basePos = *pos;
                    foundBase = true;
                    break;
                }
            }
        }

        Real dx = teamPos->x - basePos.x;
        Real dy = teamPos->y - basePos.y;
        Real dist = sqrtf(dx * dx + dy * dy);
        outState.distToBase = fminf(dist / TacticalConfig::MAX_BASE_DIST, 1.0f);
    }

    outState.underFire = data.underFire ? 1.0f : 0.0f;
    outState.hasTransport = data.hasTransport ? 1.0f : 0.0f;

    // Situational - enemy/ally quadrants
    countEnemiesInQuadrants(team, player, outState.nearbyEnemies);
    countAlliesInQuadrants(team, player, outState.nearbyAllies);

    outState.terrainAdvantage = 0.5f;  // TODO: Calculate from terrain
    outState.threatLevel = calculateThreatLevel(team, player);
    outState.targetValue = 0.5f;  // TODO: Calculate target value
    outState.supplyDist = 0.5f;   // TODO: Calculate supply distance
    outState.retreatPath = 1.0f;  // Assume clear path
    outState.reinforcePossible = 0.0f;  // TODO: Check for available reinforcements
    outState.specialReady = 0.0f;  // TODO: Check for special abilities

    // Objective info
    outState.objectiveType = 0.0f;  // TODO: Encode current objective type
    outState.objectiveX = 0.5f;
    outState.objectiveY = 0.5f;
    outState.priority = 0.5f;
    outState.progress = 0.0f;
    outState.timeOnObjective = 0.0f;

    // Temporal
    outState.timeSinceEngagement = 1.0f;  // Assume not in combat
    outState.timeSinceCommand = 0.0f;
    outState.framesSinceSpawn = 0.5f;
}

void getTeamCompositionWeights(Team* team, Real* outWeights)
{
    if (!team)
    {
        outWeights[0] = outWeights[1] = outWeights[2] = outWeights[3] = 0.25f;
        return;
    }

    Int infantry = team->countObjects(MAKE_KINDOF_MASK(KINDOF_INFANTRY), KINDOFMASK_NONE);
    Int vehicles = team->countObjects(MAKE_KINDOF_MASK(KINDOF_VEHICLE), KINDOFMASK_NONE);
    Int aircraft = team->countObjects(MAKE_KINDOF_MASK(KINDOF_AIRCRAFT), KINDOFMASK_NONE);
    Int total = infantry + vehicles + aircraft;

    if (total == 0)
    {
        outWeights[0] = outWeights[1] = outWeights[2] = outWeights[3] = 0.25f;
        return;
    }

    outWeights[0] = (Real)infantry / total;  // Infantry weight
    outWeights[1] = (Real)vehicles / total;  // Vehicle weight
    outWeights[2] = (Real)aircraft / total;  // Aircraft weight
    outWeights[3] = 0.0f;  // Mixed (remainder)
}

Real calculateTeamHealth(Team* team)
{
    if (!team)
        return 1.0f;

    TeamMemberData data;
    team->iterateObjects(collectTeamMemberData, &data);

    if (data.totalMaxHealth > 0)
        return data.totalHealth / data.totalMaxHealth;

    return 1.0f;
}

Real calculateTeamCohesion(Team* team)
{
    if (!team)
        return 1.0f;

    // Get team center
    TeamMemberData data;
    team->iterateObjects(collectTeamMemberData, &data);

    if (data.unitCount <= 1)
        return 1.0f;

    data.centerPos.x /= data.unitCount;
    data.centerPos.y /= data.unitCount;

    // Calculate max spread
    Real spreadParams[3] = { data.centerPos.x, data.centerPos.y, 0 };
    team->iterateObjects(calculateSpread, spreadParams);
    Real maxSpread = spreadParams[2];

    // Normalize: 0 spread = 1.0 cohesion, 500+ spread = 0.0 cohesion
    return 1.0f - fminf(maxSpread / 500.0f, 1.0f);
}

// Helper for quadrant counting
struct QuadrantCounter
{
    Coord3D center;
    Player* player;
    Real quadrants[4];  // NE, SE, SW, NW
    Bool countEnemies;  // true = enemies, false = allies

    QuadrantCounter() : player(NULL), countEnemies(true)
    {
        center.x = center.y = center.z = 0;
        quadrants[0] = quadrants[1] = quadrants[2] = quadrants[3] = 0;
    }
};

// Helper struct for getting first team member
struct FirstMemberFinder
{
    Object* result;
    FirstMemberFinder() : result(NULL) {}
};

static void findFirstMember(Object* obj, void* userData)
{
    FirstMemberFinder* finder = (FirstMemberFinder*)userData;
    if (!finder->result && obj && !obj->isEffectivelyDead())
        finder->result = obj;
}

// Get first living team member (needed as reference for relationship filters)
static Object* getFirstTeamMember(Team* team)
{
    if (!team)
        return NULL;

    FirstMemberFinder finder;
    team->iterateObjects(findFirstMember, &finder);
    return finder.result;
}

void countEnemiesInQuadrants(Team* team, Player* player, Real* outQuadrants)
{
    outQuadrants[0] = outQuadrants[1] = outQuadrants[2] = outQuadrants[3] = 0;

    if (!team || !player)
        return;

    const Coord3D* teamPos = team->getEstimateTeamPosition();
    if (!teamPos)
        return;

    // Need a reference object for relationship filter
    Object* refObj = getFirstTeamMember(team);
    if (!refObj)
        return;

    QuadrantCounter counter;
    counter.center = *teamPos;
    counter.player = player;
    counter.countEnemies = true;

    // Set up filters using correct API pattern
    PartitionFilterRelationship filterRel(refObj, PartitionFilterRelationship::ALLOW_ENEMIES);
    PartitionFilterAlive filterAlive;
    PartitionFilter* filters[4];
    filters[0] = &filterRel;
    filters[1] = &filterAlive;
    filters[2] = NULL;

    // Use PartitionManager to efficiently find nearby enemies
    SimpleObjectIterator* iter = ThePartitionManager->iterateObjectsInRange(
        teamPos,
        TacticalConfig::QUADRANT_RADIUS,
        FROM_CENTER_2D,
        filters
    );
    MemoryPoolObjectHolder holder(iter);

    for (Object* obj = iter->first(); obj; obj = iter->next())
    {
        if (obj->isEffectivelyDead())
            continue;

        // Skip non-combat units
        if (!obj->isKindOf(KINDOF_CAN_ATTACK))
            continue;

        const Coord3D* objPos = obj->getPosition();
        if (!objPos)
            continue;

        // Determine quadrant
        Real dx = objPos->x - counter.center.x;
        Real dy = objPos->y - counter.center.y;

        Int quadrant;
        if (dx >= 0 && dy >= 0) quadrant = 0;      // NE
        else if (dx >= 0 && dy < 0) quadrant = 1;  // SE
        else if (dx < 0 && dy < 0) quadrant = 2;   // SW
        else quadrant = 3;                          // NW

        // Weight by distance (closer = more weight)
        Real dist = sqrtf(dx * dx + dy * dy);
        Real weight = 1.0f - (dist / TacticalConfig::QUADRANT_RADIUS);
        counter.quadrants[quadrant] += weight;
    }

    // Normalize outputs to 0-1
    for (Int i = 0; i < 4; i++)
    {
        outQuadrants[i] = fminf(counter.quadrants[i] / 5.0f, 1.0f);
    }
}

void countAlliesInQuadrants(Team* team, Player* player, Real* outQuadrants)
{
    outQuadrants[0] = outQuadrants[1] = outQuadrants[2] = outQuadrants[3] = 0;

    if (!team || !player)
        return;

    const Coord3D* teamPos = team->getEstimateTeamPosition();
    if (!teamPos)
        return;

    // Need a reference object for relationship filter
    Object* refObj = getFirstTeamMember(team);
    if (!refObj)
        return;

    QuadrantCounter counter;
    counter.center = *teamPos;
    counter.player = player;
    counter.countEnemies = false;

    // Set up filters using correct API pattern
    PartitionFilterRelationship filterRel(refObj, PartitionFilterRelationship::ALLOW_ALLIES);
    PartitionFilterAlive filterAlive;
    PartitionFilter* filters[4];
    filters[0] = &filterRel;
    filters[1] = &filterAlive;
    filters[2] = NULL;

    // Use PartitionManager to efficiently find nearby allies
    SimpleObjectIterator* iter = ThePartitionManager->iterateObjectsInRange(
        teamPos,
        TacticalConfig::QUADRANT_RADIUS,
        FROM_CENTER_2D,
        filters
    );
    MemoryPoolObjectHolder holder(iter);

    for (Object* obj = iter->first(); obj; obj = iter->next())
    {
        if (obj->isEffectivelyDead())
            continue;

        // Skip non-combat units
        if (!obj->isKindOf(KINDOF_CAN_ATTACK))
            continue;

        const Coord3D* objPos = obj->getPosition();
        if (!objPos)
            continue;

        // Determine quadrant
        Real dx = objPos->x - counter.center.x;
        Real dy = objPos->y - counter.center.y;

        Int quadrant;
        if (dx >= 0 && dy >= 0) quadrant = 0;      // NE
        else if (dx >= 0 && dy < 0) quadrant = 1;  // SE
        else if (dx < 0 && dy < 0) quadrant = 2;   // SW
        else quadrant = 3;                          // NW

        // Weight by distance (closer = more weight)
        Real dist = sqrtf(dx * dx + dy * dy);
        Real weight = 1.0f - (dist / TacticalConfig::QUADRANT_RADIUS);
        counter.quadrants[quadrant] += weight;
    }

    // Normalize outputs to 0-1
    for (Int i = 0; i < 4; i++)
    {
        outQuadrants[i] = fminf(counter.quadrants[i] / 5.0f, 1.0f);
    }
}

Real calculateThreatLevel(Team* team, Player* player)
{
    if (!team || !player)
        return 0.0f;

    // Sum enemy presence in all quadrants
    Real enemies[4];
    countEnemiesInQuadrants(team, player, enemies);

    Real totalThreat = enemies[0] + enemies[1] + enemies[2] + enemies[3];
    return fminf(totalThreat / 2.0f, 1.0f);  // Normalize
}
