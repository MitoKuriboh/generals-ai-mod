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
//  Micro State - Implementation                                              //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

// MicroState.cpp
// Unit-level state for micro control decisions
// Author: Mito, 2025

#include "PreRTS.h"
#include "GameLogic/MicroState.h"
#include "GameLogic/Object.h"
#include "GameLogic/GameLogic.h"
#include "GameLogic/PartitionManager.h"
#include "GameLogic/ObjectIter.h"
#include "GameLogic/ExperienceTracker.h"
#include "GameLogic/Module/BodyModule.h"
#include "GameLogic/Module/AIUpdate.h"
#include "Common/ThingTemplate.h"

#include <cstring>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Helper Functions
// =============================================================================

// Calculate health as a percentage (0.0 to 1.0)
static Real getHealthPercent(BodyModuleInterface* body)
{
    if (!body)
        return 1.0f;
    Real maxHealth = body->getMaxHealth();
    if (maxHealth <= 0.0f)
        return 1.0f;
    return body->getHealth() / maxHealth;
}

// =============================================================================
// MicroState Implementation
// =============================================================================

void MicroState::clear()
{
    memset(this, 0, sizeof(MicroState));
}

void MicroState::toFloatArray(Real* outArray) const
{
    Int idx = 0;

    // Unit identity (4)
    outArray[idx++] = unitType;
    outArray[idx++] = isHero;
    outArray[idx++] = veterancy;
    outArray[idx++] = hasAbility;

    // Unit status (8)
    outArray[idx++] = health;
    outArray[idx++] = shield;
    outArray[idx++] = ammunition;
    outArray[idx++] = cooldown;
    outArray[idx++] = speed;
    outArray[idx++] = range;
    outArray[idx++] = dps;
    outArray[idx++] = armor;

    // Situational (12)
    outArray[idx++] = nearestEnemyDist;
    outArray[idx++] = nearestEnemyAngle;
    outArray[idx++] = nearestEnemyHealth;
    outArray[idx++] = nearestEnemyThreat;
    outArray[idx++] = nearestAllyDist;
    outArray[idx++] = inCover;
    outArray[idx++] = underFire;
    outArray[idx++] = abilityReady;
    outArray[idx++] = targetDist;
    outArray[idx++] = targetHealth;
    outArray[idx++] = targetType;
    outArray[idx++] = canRetreat;

    // Team context (4)
    outArray[idx++] = objectiveType;
    outArray[idx++] = objectiveDir;
    outArray[idx++] = teamRole;
    outArray[idx++] = priority;

    // Temporal (4)
    outArray[idx++] = timeSinceHit;
    outArray[idx++] = timeSinceShot;
    outArray[idx++] = timeInCombat;
    outArray[idx++] = movementHistory;
}

// =============================================================================
// MicroCommand Implementation
// =============================================================================

void MicroCommand::clear()
{
    action = MICRO_FOLLOW_TEAM;
    moveAngle = 0.0f;
    moveDistance = 0.0f;
    valid = FALSE;
    unitId = INVALID_ID;
}

// =============================================================================
// State Building Functions
// =============================================================================

Bool shouldMicroUnit(Object* unit)
{
    if (!unit || unit->isEffectivelyDead())
        return FALSE;

    // Check if unit is in combat
    BodyModuleInterface* body = unit->getBodyModule();
    if (body)
    {
        UnsignedInt currentFrame = TheGameLogic->getFrame();
        UnsignedInt lastDamage = body->getLastDamageTimestamp();

        // Under fire recently?
        if (currentFrame - lastDamage < 60)  // 2 seconds
            return TRUE;
    }

    // Check if unit is attacking
    AIUpdateInterface* ai = unit->getAIUpdateInterface();
    if (ai && ai->isAttacking())
        return TRUE;

    // Check if high-value unit near enemies
    const ThingTemplate* tmpl = unit->getTemplate();
    if (tmpl)
    {
        Real cost = (Real)tmpl->friend_getBuildCost();
        if (cost > MicroConfig::HIGH_VALUE_COST)
        {
            // Check for nearby enemies
            // TODO: Use PartitionManager to check for enemies in range
            return TRUE;
        }
    }

    // Check if unit has ability ready
    // TODO: Check for special abilities

    return FALSE;
}

void buildMicroState(
    Object* unit,
    const Real* teamObjective,
    MicroState& outState)
{
    outState.clear();

    if (!unit)
        return;

    const Coord3D* unitPos = unit->getPosition();
    if (!unitPos)
        return;

    // Unit identity
    outState.unitType = getUnitTypeEncoding(unit);
    outState.isHero = unit->isKindOf(KINDOF_HERO) ? 1.0f : 0.0f;

    ExperienceTracker* exp = unit->getExperienceTracker();
    if (exp)
    {
        outState.veterancy = (Real)exp->getVeterancyLevel() / 3.0f;
    }

    // TODO: Check for special abilities
    outState.hasAbility = 0.0f;

    // Unit status
    BodyModuleInterface* body = unit->getBodyModule();
    if (body)
    {
        outState.health = getHealthPercent(body);
        outState.shield = 0.0f;  // TODO: Check for shield bonuses
    }
    else
    {
        outState.health = 1.0f;
    }

    outState.ammunition = 1.0f;  // TODO: Track ammo if applicable
    outState.cooldown = 0.0f;    // TODO: Get weapon cooldown

    // Get locomotor speed
    // Note: ThingTemplate doesn't expose getMaxSpeed(), use default
    outState.speed = 0.5f;  // Default mid-speed

    // TODO: Get weapon range from actual weapons
    outState.range = 0.5f;  // Default mid-range

    outState.dps = calculateUnitDPS(unit);

    // TODO: Get armor value
    outState.armor = 0.5f;

    // Situational - find nearest enemy
    Real enemyDist = 0, enemyAngle = 0;
    Object* nearestEnemy = findNearestEnemy(unit, &enemyDist, &enemyAngle);

    if (nearestEnemy)
    {
        outState.nearestEnemyDist = fminf(enemyDist / MicroConfig::COMBAT_DETECTION_RADIUS, 1.0f);
        outState.nearestEnemyAngle = enemyAngle / (Real)M_PI;  // Normalize to -1..1

        BodyModuleInterface* enemyBody = nearestEnemy->getBodyModule();
        if (enemyBody)
        {
            outState.nearestEnemyHealth = getHealthPercent(enemyBody);
        }
        else
        {
            outState.nearestEnemyHealth = 1.0f;
        }

        // Calculate threat based on enemy DPS vs our health
        outState.nearestEnemyThreat = calculateUnitDPS(nearestEnemy);
    }
    else
    {
        outState.nearestEnemyDist = 1.0f;
        outState.nearestEnemyAngle = 0.0f;
        outState.nearestEnemyHealth = 0.0f;
        outState.nearestEnemyThreat = 0.0f;
    }

    // TODO: Find nearest ally
    outState.nearestAllyDist = 0.5f;

    // Check if in cover/garrison
    // Note: No direct KINDOF for garrisoned state, would need to check containment
    outState.inCover = 0.0f;  // TODO: Check if unit is inside a garrison

    // Check if under fire
    if (body)
    {
        UnsignedInt currentFrame = TheGameLogic->getFrame();
        UnsignedInt lastDamage = body->getLastDamageTimestamp();
        outState.underFire = (currentFrame - lastDamage < 30) ? 1.0f : 0.0f;
    }

    // TODO: Check ability cooldown
    outState.abilityReady = 0.0f;

    // Current target info
    AIUpdateInterface* ai = unit->getAIUpdateInterface();
    if (ai)
    {
        Object* target = ai->getCurrentVictim();
        if (target)
        {
            const Coord3D* targetPos = target->getPosition();
            if (targetPos)
            {
                Real dx = targetPos->x - unitPos->x;
                Real dy = targetPos->y - unitPos->y;
                Real dist = sqrtf(dx * dx + dy * dy);
                outState.targetDist = fminf(dist / MicroConfig::COMBAT_DETECTION_RADIUS, 1.0f);
            }

            BodyModuleInterface* targetBody = target->getBodyModule();
            if (targetBody)
            {
                outState.targetHealth = getHealthPercent(targetBody);
            }

            outState.targetType = getUnitTypeEncoding(target);
        }
    }

    // TODO: Check retreat path
    outState.canRetreat = 1.0f;

    // Team context
    if (teamObjective)
    {
        outState.objectiveType = teamObjective[0];
        outState.objectiveDir = teamObjective[1];
        outState.teamRole = teamObjective[2];
        outState.priority = teamObjective[3];
    }
    else
    {
        outState.objectiveType = 0.0f;
        outState.objectiveDir = 0.0f;
        outState.teamRole = 0.0f;
        outState.priority = 0.5f;
    }

    // Temporal
    if (body)
    {
        UnsignedInt currentFrame = TheGameLogic->getFrame();
        UnsignedInt lastDamage = body->getLastDamageTimestamp();
        Real framesSinceHit = (Real)(currentFrame - lastDamage);
        outState.timeSinceHit = fminf(framesSinceHit / 300.0f, 1.0f);  // Normalize to 10 seconds
    }
    else
    {
        outState.timeSinceHit = 1.0f;
    }

    // TODO: Track time since shot
    outState.timeSinceShot = 0.0f;

    // TODO: Track combat duration
    outState.timeInCombat = 0.5f;

    // TODO: Track movement history
    outState.movementHistory = 0.0f;
}

Object* findNearestEnemy(Object* unit, Real* outDist, Real* outAngle)
{
    if (!unit)
    {
        if (outDist) *outDist = 1000000.0f;
        if (outAngle) *outAngle = 0.0f;
        return NULL;
    }

    const Coord3D* unitPos = unit->getPosition();
    if (!unitPos)
        return NULL;

    Player* player = unit->getControllingPlayer();
    if (!player)
        return NULL;

    Object* nearestEnemy = NULL;
    Real nearestDist = MicroConfig::COMBAT_DETECTION_RADIUS;

    // Set up filters using correct API pattern
    PartitionFilterRelationship filterRel(unit, PartitionFilterRelationship::ALLOW_ENEMIES);
    PartitionFilterAlive filterAlive;
    PartitionFilter* filters[4];
    filters[0] = &filterRel;
    filters[1] = &filterAlive;
    filters[2] = NULL;

    // Use PartitionManager to find nearby objects
    SimpleObjectIterator* iter = ThePartitionManager->iterateObjectsInRange(
        unitPos,
        MicroConfig::COMBAT_DETECTION_RADIUS,
        FROM_CENTER_2D,
        filters
    );
    MemoryPoolObjectHolder holder(iter);

    for (Object* obj = iter->first(); obj; obj = iter->next())
    {
        if (obj->isEffectivelyDead())
            continue;

        const Coord3D* objPos = obj->getPosition();
        if (!objPos)
            continue;

        Real dx = objPos->x - unitPos->x;
        Real dy = objPos->y - unitPos->y;
        Real dist = sqrtf(dx * dx + dy * dy);

        if (dist < nearestDist)
        {
            nearestDist = dist;
            nearestEnemy = obj;
        }
    }

    if (nearestEnemy && outAngle)
    {
        const Coord3D* enemyPos = nearestEnemy->getPosition();
        Real dx = enemyPos->x - unitPos->x;
        Real dy = enemyPos->y - unitPos->y;
        *outAngle = atan2f(dy, dx);
    }

    if (outDist)
        *outDist = nearestDist;

    return nearestEnemy;
}

Object* findWeakestEnemy(Object* unit, Real range)
{
    if (!unit)
        return NULL;

    const Coord3D* unitPos = unit->getPosition();
    if (!unitPos)
        return NULL;

    Player* player = unit->getControllingPlayer();
    if (!player)
        return NULL;

    Object* weakestEnemy = NULL;
    Real lowestHealth = 2.0f;  // Impossible value

    // Set up filters using correct API pattern
    PartitionFilterRelationship filterRel(unit, PartitionFilterRelationship::ALLOW_ENEMIES);
    PartitionFilterAlive filterAlive;
    PartitionFilter* filters[4];
    filters[0] = &filterRel;
    filters[1] = &filterAlive;
    filters[2] = NULL;

    SimpleObjectIterator* iter = ThePartitionManager->iterateObjectsInRange(
        unitPos,
        range,
        FROM_CENTER_2D,
        filters
    );
    MemoryPoolObjectHolder holder(iter);

    for (Object* obj = iter->first(); obj; obj = iter->next())
    {
        if (obj->isEffectivelyDead())
            continue;

        BodyModuleInterface* body = obj->getBodyModule();
        if (!body)
            continue;

        Real health = getHealthPercent(body);
        if (health < lowestHealth)
        {
            lowestHealth = health;
            weakestEnemy = obj;
        }
    }

    return weakestEnemy;
}

Object* findPriorityTarget(Object* unit, Real range)
{
    if (!unit)
        return NULL;

    const Coord3D* unitPos = unit->getPosition();
    if (!unitPos)
        return NULL;

    Player* player = unit->getControllingPlayer();
    if (!player)
        return NULL;

    Object* priorityTarget = NULL;
    Real highestValue = 0.0f;

    // Set up filters using correct API pattern
    PartitionFilterRelationship filterRel(unit, PartitionFilterRelationship::ALLOW_ENEMIES);
    PartitionFilterAlive filterAlive;
    PartitionFilter* filters[4];
    filters[0] = &filterRel;
    filters[1] = &filterAlive;
    filters[2] = NULL;

    SimpleObjectIterator* iter = ThePartitionManager->iterateObjectsInRange(
        unitPos,
        range,
        FROM_CENTER_2D,
        filters
    );
    MemoryPoolObjectHolder holder(iter);

    for (Object* obj = iter->first(); obj; obj = iter->next())
    {
        if (obj->isEffectivelyDead())
            continue;

        const ThingTemplate* tmpl = obj->getTemplate();
        if (!tmpl)
            continue;

        // Priority based on: cost + hero bonus
        Real value = (Real)tmpl->friend_getBuildCost();
        if (obj->isKindOf(KINDOF_HERO))
            value *= 3.0f;

        if (value > highestValue)
        {
            highestValue = value;
            priorityTarget = obj;
        }
    }

    return priorityTarget;
}

void calculateKitePosition(Object* unit, const Coord3D* targetPos, Coord3D* outPos)
{
    if (!unit || !targetPos || !outPos)
        return;

    const Coord3D* unitPos = unit->getPosition();
    if (!unitPos)
        return;

    // Calculate direction away from target
    Real dx = unitPos->x - targetPos->x;
    Real dy = unitPos->y - targetPos->y;
    Real dist = sqrtf(dx * dx + dy * dy);

    if (dist < 0.001f)
    {
        // Target on top of us, pick random direction
        dx = 1.0f;
        dy = 0.0f;
        dist = 1.0f;
    }

    // Normalize direction
    dx /= dist;
    dy /= dist;

    // Get weapon range (TODO: Get actual range from weapon)
    Real weaponRange = 300.0f;  // Default range
    Real desiredDist = weaponRange * MicroConfig::KITE_RANGE_FACTOR;

    // Move back to maintain desired distance
    Real moveAmount = desiredDist - dist;
    if (moveAmount > 0)
        moveAmount = fminf(moveAmount, 50.0f);  // Cap movement per step

    outPos->x = unitPos->x + dx * moveAmount;
    outPos->y = unitPos->y + dy * moveAmount;
    outPos->z = unitPos->z;
}

void calculateFlankPosition(Object* unit, const Coord3D* targetPos, Bool clockwise, Coord3D* outPos)
{
    if (!unit || !targetPos || !outPos)
        return;

    const Coord3D* unitPos = unit->getPosition();
    if (!unitPos)
        return;

    // Calculate current angle to target
    Real dx = unitPos->x - targetPos->x;
    Real dy = unitPos->y - targetPos->y;
    Real dist = sqrtf(dx * dx + dy * dy);

    if (dist < 0.001f)
    {
        *outPos = *unitPos;
        return;
    }

    Real currentAngle = atan2f(dy, dx);

    // Rotate angle by 30 degrees (pi/6) in chosen direction
    Real rotateAmount = (Real)M_PI / 6.0f;
    if (!clockwise)
        rotateAmount = -rotateAmount;

    Real newAngle = currentAngle + rotateAmount;

    // Keep same distance from target
    outPos->x = targetPos->x + cosf(newAngle) * dist;
    outPos->y = targetPos->y + sinf(newAngle) * dist;
    outPos->z = unitPos->z;
}

Real checkRetreatPath(Object* unit, const Coord3D* basePos)
{
    if (!unit || !basePos)
        return 1.0f;

    const Coord3D* unitPos = unit->getPosition();
    if (!unitPos)
        return 1.0f;

    Player* player = unit->getControllingPlayer();
    if (!player)
        return 1.0f;

    // Check for enemies between unit and base
    Real dx = basePos->x - unitPos->x;
    Real dy = basePos->y - unitPos->y;
    Real dist = sqrtf(dx * dx + dy * dy);

    if (dist < 100.0f)
        return 1.0f;  // Already near base

    // Normalize direction
    dx /= dist;
    dy /= dist;

    // Set up filters using correct API pattern
    PartitionFilterRelationship filterRel(unit, PartitionFilterRelationship::ALLOW_ENEMIES);
    PartitionFilterAlive filterAlive;
    PartitionFilter* filters[4];
    filters[0] = &filterRel;
    filters[1] = &filterAlive;
    filters[2] = NULL;

    // Check several points along retreat path
    Real clearness = 1.0f;
    for (Real t = 0.25f; t <= 0.75f; t += 0.25f)
    {
        Coord3D checkPos;
        checkPos.x = unitPos->x + dx * dist * t;
        checkPos.y = unitPos->y + dy * dist * t;
        checkPos.z = unitPos->z;

        // Count enemies near this point
        SimpleObjectIterator* iter = ThePartitionManager->iterateObjectsInRange(
            &checkPos,
            150.0f,
            FROM_CENTER_2D,
            filters
        );
        MemoryPoolObjectHolder holder(iter);

        Int enemyCount = 0;
        for (Object* obj = iter->first(); obj; obj = iter->next())
        {
            if (!obj->isEffectivelyDead())
                enemyCount++;
        }

        // Reduce clearness based on enemy presence
        clearness -= (Real)enemyCount * 0.1f;
    }

    return fmaxf(clearness, 0.0f);
}

Real getUnitTypeEncoding(Object* unit)
{
    if (!unit)
        return 0.0f;

    // Encode unit type as normalized value
    // Infantry: 0.0-0.3
    // Vehicles: 0.3-0.6
    // Aircraft: 0.6-0.9
    // Structures: 0.9-1.0

    if (unit->isKindOf(KINDOF_INFANTRY))
    {
        if (unit->isKindOf(KINDOF_HERO))
            return 0.25f;
        else
            return 0.1f;
    }
    else if (unit->isKindOf(KINDOF_VEHICLE))
    {
        if (unit->isKindOf(KINDOF_HUGE_VEHICLE))
            return 0.5f;  // Heavy vehicles (Overlord, etc.)
        else if (unit->isKindOf(KINDOF_TRANSPORT))
            return 0.4f;
        else
            return 0.35f;
    }
    else if (unit->isKindOf(KINDOF_AIRCRAFT))
    {
        return 0.7f;
    }
    else if (unit->isKindOf(KINDOF_STRUCTURE))
    {
        return 0.95f;
    }

    return 0.5f;  // Unknown
}

Real calculateUnitDPS(Object* unit)
{
    if (!unit)
        return 0.0f;

    // TODO: Get actual weapon data and calculate DPS
    // For now, estimate based on unit type and cost

    const ThingTemplate* tmpl = unit->getTemplate();
    if (!tmpl)
        return 0.0f;

    Real cost = (Real)tmpl->friend_getBuildCost();

    // Rough DPS estimate: cost / 100, capped at 1.0
    Real dps = fminf(cost / 1000.0f, 1.0f);

    // Bonus for heavy combat units
    if (unit->isKindOf(KINDOF_HUGE_VEHICLE) || unit->isKindOf(KINDOF_AIRCRAFT))
        dps *= 1.5f;

    return fminf(dps, 1.0f);
}
