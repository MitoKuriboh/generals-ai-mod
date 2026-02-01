/*
**	Command & Conquer Generals Zero Hour(tm)
**	Copyright 2025 Electronic Arts Inc.
**
**	This program is free software: you can redistribute it and/or modify
**	it under the terms of the GNU General Public License as published by
**	the Free Software Foundation, either version 3 of the License, or
**	(at your option) any later version.
*/

////////////////////////////////////////////////////////////////////////////////
//  Learning AI Player - ML-driven strategic decisions                        //
////////////////////////////////////////////////////////////////////////////////

#include "PreRTS.h"

#include <math.h>
#include <vector>
#include <stdio.h>
#include <stdarg.h>
#include "Common/GameMemory.h"
#include "Common/GlobalData.h"
#include "Common/Player.h"
#include "Common/PlayerList.h"
#include "Common/PlayerTemplate.h"
#include "Common/Xfer.h"
#include "Common/ThingTemplate.h"
#include "Common/ThingFactory.h"
#include "Common/Team.h"
#include "GameLogic/GameLogic.h"
#include "GameLogic/Object.h"
#include "GameLogic/AILearningPlayer.h"
#include "GameLogic/AI.h"
#include "GameLogic/Module/BodyModule.h"
#include "GameLogic/SidesList.h"  // For BuildListInfo
#include "GameLogic/ScriptEngine.h"
#include "GameLogic/LogicRandomValue.h"
#include "GameLogic/TacticalState.h"
#include "GameLogic/MicroState.h"
#include "GameLogic/Module/AIUpdate.h"

// File-based ML logging (works in release builds)
#define ENABLE_ML_FILE_LOG 1

#ifdef ENABLE_ML_FILE_LOG
static FILE* g_mlLogFile = NULL;
static void mlLogWrite(const char* fmt, ...) {
	if (!g_mlLogFile) {
		g_mlLogFile = fopen("C:\\Users\\Public\\ml_decisions.log", "a");
	}
	if (g_mlLogFile) {
		va_list args;
		va_start(args, fmt);
		vfprintf(g_mlLogFile, fmt, args);
		va_end(args);
		fflush(g_mlLogFile);
	}
}
#define ML_LOG(x) mlLogWrite x
#else
#define ML_LOG(x)
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// CONSTRUCTOR / DESTRUCTOR
///////////////////////////////////////////////////////////////////////////////////////////////////

AILearningPlayer::AILearningPlayer( Player *p ) : AISkirmishPlayer(p),
	m_frameCounter(0),
	m_lastFrameMoney(0.0f),
	m_recentDamageTaken(0.0f),
	m_lastDamageFrame(0),
	m_teamsHeld(0),
	m_lastAttackFrame(0),
	m_gameEndSent(false),
	m_tacticalEnabled(true),
	m_tacticalDecisionInterval(TacticalConfig::DECISION_INTERVAL),
	m_microEnabled(true),
	m_microDecisionInterval(MicroConfig::DECISION_INTERVAL),
	m_trackedUnitCount(0),
	m_hasValidBasePos(false)
{
	m_currentRecommendation.clear();

	// Initialize tracking arrays
	memset(m_lastTacticalFrame, 0, sizeof(m_lastTacticalFrame));
	memset(m_lastMicroFrame, 0, sizeof(m_lastMicroFrame));
	memset(m_trackedUnitIds, 0, sizeof(m_trackedUnitIds));
	memset(m_strategicOutput, 0, sizeof(m_strategicOutput));
	m_cachedBasePos.x = m_cachedBasePos.y = m_cachedBasePos.z = 0;

	DEBUG_LOG(("AILearningPlayer created for player %s\n",
		TheNameKeyGenerator->keyToName(p->getPlayerNameKey()).str()));
}

AILearningPlayer::~AILearningPlayer()
{
	m_mlBridge.disconnect();
	DEBUG_LOG(("AILearningPlayer destroyed\n"));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// MAIN UPDATE LOOP
///////////////////////////////////////////////////////////////////////////////////////////////////

void AILearningPlayer::update()
{
	if (!m_player) return;

	m_frameCounter++;

	// Check for game end
	checkGameEnd();

	// Communicate with ML bridge at configured interval
	if (m_frameCounter % MLConfig::DECISION_INTERVAL == 0) {
		exportStateToML();
		processMLRecommendations();

		// Cache strategic output for tactical layer
		if (m_currentRecommendation.valid) {
			m_strategicOutput[0] = m_currentRecommendation.priorityEconomy;
			m_strategicOutput[1] = m_currentRecommendation.priorityDefense;
			m_strategicOutput[2] = m_currentRecommendation.priorityMilitary;
			m_strategicOutput[3] = m_currentRecommendation.priorityTech;
			m_strategicOutput[4] = m_currentRecommendation.preferInfantry;
			m_strategicOutput[5] = m_currentRecommendation.preferVehicles;
			m_strategicOutput[6] = m_currentRecommendation.preferAircraft;
			m_strategicOutput[7] = m_currentRecommendation.aggression;
		}
	}

	// Process tactical layer for team-level decisions
	// Only if: local config allows it, batched mode is enabled, AND server has tactical capability
	if (m_tacticalEnabled && m_mlBridge.isBatchedModeEnabled() &&
		m_mlBridge.getCapabilities().tactical) {
		processTeamTactics();
	}

	// Process micro layer for unit-level decisions
	// Only if: local config allows it, batched mode is enabled, AND server has micro capability
	if (m_microEnabled && m_mlBridge.isBatchedModeEnabled() &&
		m_mlBridge.getCapabilities().micro) {
		processMicroControl();
	}

	// Run normal skirmish AI
	AISkirmishPlayer::update();
}

void AILearningPlayer::newMap()
{
	m_frameCounter = 0;
	m_lastFrameMoney = 0.0f;
	m_recentDamageTaken = 0.0f;
	m_lastDamageFrame = 0;
	m_teamsHeld = 0;
	m_lastAttackFrame = 0;
	m_gameEndSent = false;
	m_currentRecommendation.clear();

	// Reset hierarchical tracking
	memset(m_lastTacticalFrame, 0, sizeof(m_lastTacticalFrame));
	memset(m_lastMicroFrame, 0, sizeof(m_lastMicroFrame));
	memset(m_trackedUnitIds, 0, sizeof(m_trackedUnitIds));
	memset(m_strategicOutput, 0, sizeof(m_strategicOutput));
	m_trackedUnitCount = 0;
	m_hasValidBasePos = false;
	m_cachedBasePos.x = m_cachedBasePos.y = m_cachedBasePos.z = 0;

	m_mlBridge.connect();
	AISkirmishPlayer::newMap();
}

void AILearningPlayer::onUnitProduced( Object *factory, Object *unit )
{
	AISkirmishPlayer::onUnitProduced(factory, unit);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// ML BRIDGE COMMUNICATION
///////////////////////////////////////////////////////////////////////////////////////////////////

void AILearningPlayer::exportStateToML()
{
	MLGameState state;
	buildGameState(state);

	ML_LOG(("MLState: frame=%d player=%d money=%.2f\n",
		m_frameCounter, state.playerIndex, state.money));

	m_mlBridge.sendState(state);
}

void AILearningPlayer::processMLRecommendations()
{
	MLRecommendation rec;
	if (m_mlBridge.receiveRecommendation(rec)) {
		m_currentRecommendation = rec;
		ML_LOG(("MLBridge: aggr=%.2f eco=%.2f mil=%.2f\n",
			rec.aggression, rec.priorityEconomy, rec.priorityMilitary));
	} else if (m_mlBridge.isRecommendationStale()) {
		// Recommendation is stale - revert to defaults
		m_currentRecommendation = m_mlBridge.getValidRecommendation();
		if (!m_currentRecommendation.valid) {
			ML_LOG(("MLBridge: Recommendation stale, using parent AI logic\n"));
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// STATE EXTRACTION - Simplified version
///////////////////////////////////////////////////////////////////////////////////////////////////

void AILearningPlayer::buildGameState(MLGameState& state)
{
	state.clear();
	if (!m_player) return;

	state.playerIndex = m_player->getPlayerIndex();

	// Economy
	Money* playerMoney = m_player->getMoney();
	UnsignedInt money = playerMoney ? playerMoney->countMoney() : 0;
	state.money = (Real)log10((double)(money + 1));

	// Income rate (calculated from frame-to-frame money change)
	state.incomeRate = calculateIncomeRate();

	// Power
	Energy* energy = m_player->getEnergy();
	if (energy) {
		state.powerBalance = (Real)(energy->getProduction() - energy->getConsumption());
	}

	// Supply usage
	state.supplyUsed = calculateSupplyUsed();

	// Count forces
	countForces(state.ownInfantry, state.ownVehicles, state.ownAircraft, state.ownStructures, true);
	countForces(state.enemyInfantry, state.enemyVehicles, state.enemyAircraft, state.enemyStructures, false);

	// Time and metrics
	state.gameTimeMinutes = (Real)TheGameLogic->getFrame() / (30.0f * 60.0f);
	state.techLevel = calculateTechLevel();
	state.baseThreat = calculateBaseThreat();
	state.armyStrength = calculateArmyStrength();
	state.distanceToEnemy = calculateDistanceToEnemy();
	state.underAttack = calculateUnderAttack();

	// Faction detection (one-hot encoded)
	const PlayerTemplate* playerTemplate = m_player->getPlayerTemplate();
	if (playerTemplate) {
		const char* factionName = playerTemplate->getName().str();
		// Check for faction substring in player template name
		if (strstr(factionName, "America") != NULL || strstr(factionName, "USA") != NULL) {
			state.isUSA = 1.0f;
		} else if (strstr(factionName, "China") != NULL) {
			state.isChina = 1.0f;
		} else if (strstr(factionName, "GLA") != NULL) {
			state.isGLA = 1.0f;
		}
	}
}

void AILearningPlayer::countForces(Real* infantry, Real* vehicles, Real* aircraft, Real* structures, Bool own)
{
	// Array format: [0] = log10(count+1), [1] = avg health ratio (0-1), [2] = in production count
	infantry[0] = infantry[1] = infantry[2] = 0.0f;
	vehicles[0] = vehicles[1] = vehicles[2] = 0.0f;
	aircraft[0] = aircraft[1] = aircraft[2] = 0.0f;
	structures[0] = structures[1] = structures[2] = 0.0f;

	Int infantryCount = 0, vehicleCount = 0, aircraftCount = 0, structureCount = 0;
	Real infantryHealth = 0.0f, vehicleHealth = 0.0f, aircraftHealth = 0.0f, structureHealth = 0.0f;

	for (Object* obj = TheGameLogic->getFirstObject(); obj; obj = obj->getNextObject()) {
		if (obj->isEffectivelyDead()) continue;

		Bool isOwn = (obj->getControllingPlayer() == m_player);
		if (isOwn != own) continue;

		// Calculate health ratio for this unit
		Real healthRatio = 1.0f;
		const BodyModuleInterface* body = obj->getBodyModule();
		if (body) {
			Real maxHealth = body->getMaxHealth();
			if (maxHealth > 0) {
				healthRatio = body->getHealth() / maxHealth;
			}
		}

		if (obj->isKindOf(KINDOF_INFANTRY)) {
			infantryCount++;
			infantryHealth += healthRatio;
		} else if (obj->isKindOf(KINDOF_VEHICLE)) {
			vehicleCount++;
			vehicleHealth += healthRatio;
		} else if (obj->isKindOf(KINDOF_AIRCRAFT)) {
			aircraftCount++;
			aircraftHealth += healthRatio;
		} else if (obj->isKindOf(KINDOF_STRUCTURE)) {
			structureCount++;
			structureHealth += healthRatio;
		}
	}

	// [0] = log10(count+1) for scale-invariant count
	infantry[0] = (Real)log10((double)(infantryCount + 1));
	vehicles[0] = (Real)log10((double)(vehicleCount + 1));
	aircraft[0] = (Real)log10((double)(aircraftCount + 1));
	structures[0] = (Real)log10((double)(structureCount + 1));

	// [1] = average health ratio (0-1)
	infantry[1] = (infantryCount > 0) ? (infantryHealth / infantryCount) : 0.0f;
	vehicles[1] = (vehicleCount > 0) ? (vehicleHealth / vehicleCount) : 0.0f;
	aircraft[1] = (aircraftCount > 0) ? (aircraftHealth / aircraftCount) : 0.0f;
	structures[1] = (structureCount > 0) ? (structureHealth / structureCount) : 0.0f;

	// [2] = production queue count (only for own units)
	// Note: Production tracking would require accessing production queue data
	// which isn't easily accessible here. Leave as 0 for now - can be enhanced later
	// when we have access to the production system's queue state.
}

Real AILearningPlayer::calculateTechLevel()
{
	Int techBuildings = 0;
	for (Object* obj = TheGameLogic->getFirstObject(); obj; obj = obj->getNextObject()) {
		if (obj->getControllingPlayer() != m_player) continue;
		if (obj->isEffectivelyDead()) continue;
		if (obj->isKindOf(KINDOF_FS_STRATEGY_CENTER) ||
			obj->isKindOf(KINDOF_FS_ADVANCED_TECH) ||
			obj->isKindOf(KINDOF_FS_TECHNOLOGY)) {
			techBuildings++;
		}
	}
	return (Real)techBuildings / 5.0f;
}

Real AILearningPlayer::calculateBaseThreat()
{
	Coord3D basePos = {0, 0, 0};
	Bool hasBase = false;

	for (Object* obj = TheGameLogic->getFirstObject(); obj; obj = obj->getNextObject()) {
		if (obj->getControllingPlayer() != m_player) continue;
		if (obj->isKindOf(KINDOF_COMMANDCENTER) && !obj->isEffectivelyDead()) {
			const Coord3D* baseCoord = obj->getPosition();
			if (!baseCoord) continue;
			basePos = *baseCoord;
			hasBase = true;
			break;
		}
	}

	if (!hasBase) return 1.0f;

	Int nearbyEnemies = 0;

	for (Object* obj = TheGameLogic->getFirstObject(); obj; obj = obj->getNextObject()) {
		if (obj->getControllingPlayer() == m_player) continue;
		if (obj->isEffectivelyDead()) continue;
		if (!obj->isKindOf(KINDOF_CAN_ATTACK)) continue;

		const Coord3D* pos = obj->getPosition();
		if (!pos) continue;
		Real dx = pos->x - basePos.x;
		Real dy = pos->y - basePos.y;
		Real dist = sqrt(dx*dx + dy*dy);

		if (dist < MLConfig::THREAT_DETECTION_RADIUS) {
			nearbyEnemies++;
		}
	}

	return min(1.0f, (Real)nearbyEnemies / 20.0f);
}

Real AILearningPlayer::calculateArmyStrength()
{
	Int ownUnits = 0, enemyUnits = 0;

	for (Object* obj = TheGameLogic->getFirstObject(); obj; obj = obj->getNextObject()) {
		if (obj->isEffectivelyDead()) continue;
		if (!obj->isKindOf(KINDOF_CAN_ATTACK)) continue;

		if (obj->getControllingPlayer() == m_player) {
			ownUnits++;
		} else {
			// Check if enemy (simplified - count all non-own units)
			Player* otherPlayer = obj->getControllingPlayer();
			if (otherPlayer && m_player->getRelationship(otherPlayer->getDefaultTeam()) == ENEMIES) {
				enemyUnits++;
			}
		}
	}

	if (enemyUnits < 1) return MLConfig::MAX_ARMY_STRENGTH_RATIO;
	return min(MLConfig::MAX_ARMY_STRENGTH_RATIO, (Real)ownUnits / (Real)enemyUnits);
}

Real AILearningPlayer::calculateDistanceToEnemy()
{
	Coord3D myBase = {0, 0, 0};
	Coord3D enemyBase = {0, 0, 0};
	Bool foundMy = false, foundEnemy = false;

	for (Object* obj = TheGameLogic->getFirstObject(); obj; obj = obj->getNextObject()) {
		if (!obj->isKindOf(KINDOF_COMMANDCENTER)) continue;
		if (obj->isEffectivelyDead()) continue;

		const Coord3D* pos = obj->getPosition();
		if (!pos) continue;

		if (obj->getControllingPlayer() == m_player) {
			myBase = *pos;
			foundMy = true;
		} else {
			Player* otherPlayer = obj->getControllingPlayer();
			if (otherPlayer && m_player->getRelationship(otherPlayer->getDefaultTeam()) == ENEMIES) {
				enemyBase = *pos;
				foundEnemy = true;
			}
		}
	}

	if (!foundMy || !foundEnemy) return 1.0f;

	Real dx = enemyBase.x - myBase.x;
	Real dy = enemyBase.y - myBase.y;
	return min(1.0f, sqrt(dx*dx + dy*dy) / MLConfig::NORMALIZED_MAP_SCALE);
}

Real AILearningPlayer::calculateIncomeRate()
{
	if (!m_player) return 0.0f;

	Money* playerMoney = m_player->getMoney();
	Real currentMoney = playerMoney ? (Real)playerMoney->countMoney() : 0.0f;
	Real income = 0.0f;

	// Calculate income as money delta per second (30 logic frames = 1 second)
	if (m_lastFrameMoney > 0) {
		income = (currentMoney - m_lastFrameMoney) * LOGICFRAMES_PER_SECOND / MLConfig::DECISION_INTERVAL;
	}

	m_lastFrameMoney = currentMoney;

	// Normalize to reasonable range (typical income is 0-1000 per second)
	return income / 100.0f;
}

Real AILearningPlayer::calculateSupplyUsed()
{
	// Supply usage isn't directly exposed in the game API
	// Approximate based on unit counts vs typical supply limits
	Int unitCount = 0;
	const Int maxSupply = 100;  // Typical supply cap

	for (Object* obj = TheGameLogic->getFirstObject(); obj; obj = obj->getNextObject()) {
		if (obj->getControllingPlayer() != m_player) continue;
		if (obj->isEffectivelyDead()) continue;
		if (obj->isKindOf(KINDOF_INFANTRY) || obj->isKindOf(KINDOF_VEHICLE) || obj->isKindOf(KINDOF_AIRCRAFT)) {
			unitCount++;
		}
	}

	return min(1.0f, (Real)unitCount / (Real)maxSupply);
}

Real AILearningPlayer::calculateUnderAttack()
{
	// Check if any of our structures are being attacked
	// This is approximated by checking base threat level
	Real threat = calculateBaseThreat();

	// Also check if we've taken damage recently (within 10 seconds)
	// Increased from 5 seconds to catch faster attack patterns
	UnsignedInt currentFrame = TheGameLogic->getFrame();
	UnsignedInt recentWindow = 10 * LOGICFRAMES_PER_SECOND;

	Bool recentlyDamaged = (currentFrame - m_lastDamageFrame) < recentWindow;

	if (threat > 0.3f || recentlyDamaged) {
		return 1.0f;
	}
	return 0.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// TEAM CLASSIFICATION - Analyzes team template units
///////////////////////////////////////////////////////////////////////////////////////////////////

TeamCategory AILearningPlayer::classifyTeam(TeamPrototype* proto)
{
	if (!proto) return TEAM_CATEGORY_MIXED;

	const TeamTemplateInfo* info = proto->getTemplateInfo();
	if (!info) return TEAM_CATEGORY_MIXED;

	Int infantry = 0, vehicles = 0, aircraft = 0;

	// Iterate through unit types in the team template
	for (Int i = 0; i < info->m_numUnitsInfo; i++) {
		const TCreateUnitsInfo& unitInfo = info->m_unitsInfo[i];
		const ThingTemplate* tmpl = TheThingFactory->findTemplate(unitInfo.unitThingName);
		if (!tmpl) continue;

		Int count = unitInfo.maxUnits;
		if (tmpl->isKindOf(KINDOF_INFANTRY)) {
			infantry += count;
		} else if (tmpl->isKindOf(KINDOF_AIRCRAFT)) {
			aircraft += count;
		} else if (tmpl->isKindOf(KINDOF_VEHICLE)) {
			vehicles += count;
		}
	}

	// Return dominant category or mixed if balanced
	Int total = infantry + vehicles + aircraft;
	if (total == 0) return TEAM_CATEGORY_MIXED;

	// 60% threshold for dominant category
	if (infantry * 10 > total * 6) return TEAM_CATEGORY_INFANTRY;
	if (vehicles * 10 > total * 6) return TEAM_CATEGORY_VEHICLE;
	if (aircraft * 10 > total * 6) return TEAM_CATEGORY_AIRCRAFT;

	return TEAM_CATEGORY_MIXED;
}

Real AILearningPlayer::getTeamCategoryWeight(TeamCategory category)
{
	if (!m_currentRecommendation.valid) return 1.0f;

	switch (category) {
		case TEAM_CATEGORY_INFANTRY: return m_currentRecommendation.preferInfantry;
		case TEAM_CATEGORY_VEHICLE: return m_currentRecommendation.preferVehicles;
		case TEAM_CATEGORY_AIRCRAFT: return m_currentRecommendation.preferAircraft;
		default: return 1.0f;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// BUILDING CLASSIFICATION
///////////////////////////////////////////////////////////////////////////////////////////////////

BuildingCategory AILearningPlayer::classifyBuilding(const ThingTemplate* tmpl)
{
	if (!tmpl) return BUILDING_CATEGORY_UNKNOWN;

	if (tmpl->isKindOf(KINDOF_FS_SUPPLY_CENTER) || tmpl->isKindOf(KINDOF_FS_SUPPLY_DROPZONE)) {
		return BUILDING_CATEGORY_ECONOMY;
	}
	if (tmpl->isKindOf(KINDOF_FS_POWER)) {
		return BUILDING_CATEGORY_POWER;
	}
	if (tmpl->isKindOf(KINDOF_FS_BASE_DEFENSE)) {
		return BUILDING_CATEGORY_DEFENSE;
	}
	if (tmpl->isKindOf(KINDOF_FS_STRATEGY_CENTER) || tmpl->isKindOf(KINDOF_FS_TECHNOLOGY)) {
		return BUILDING_CATEGORY_TECH;
	}
	if (tmpl->isKindOf(KINDOF_FS_SUPERWEAPON)) {
		return BUILDING_CATEGORY_SUPER;
	}
	if (tmpl->isKindOf(KINDOF_FS_FACTORY) || tmpl->isKindOf(KINDOF_FS_BARRACKS)) {
		return BUILDING_CATEGORY_MILITARY;
	}

	return BUILDING_CATEGORY_UNKNOWN;
}

Real AILearningPlayer::getBuildingCategoryWeight(BuildingCategory category)
{
	if (!m_currentRecommendation.valid) return 1.0f;

	switch (category) {
		case BUILDING_CATEGORY_ECONOMY:
		case BUILDING_CATEGORY_POWER:
			return m_currentRecommendation.priorityEconomy;
		case BUILDING_CATEGORY_DEFENSE:
			return m_currentRecommendation.priorityDefense;
		case BUILDING_CATEGORY_MILITARY:
			return m_currentRecommendation.priorityMilitary;
		case BUILDING_CATEGORY_TECH:
		case BUILDING_CATEGORY_SUPER:
			return m_currentRecommendation.priorityTech;
		default:
			return 1.0f;
	}
}

Bool AILearningPlayer::shouldDelayBuilding(BuildingCategory category)
{
	if (!m_currentRecommendation.valid) return false;

	// CRITICAL: Never delay economy buildings when low on money
	// This prevents the AI from starving itself to death
	if (category == BUILDING_CATEGORY_ECONOMY) {
		Money* money = m_player->getMoney();
		UnsignedInt currentMoney = money ? money->countMoney() : 0;
		if (currentMoney < 1000) {
			return false;  // Force economy building when broke
		}
	}

	Real weight = getBuildingCategoryWeight(category);
	return (weight < MLConfig::DELAY_THRESHOLD);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// ATTACK TIMING
///////////////////////////////////////////////////////////////////////////////////////////////////

// NOTE: shouldHoldTeam() removed as dead code - attack timing handled by checkReadyTeams()

Int AILearningPlayer::getMinArmySizeForAttack()
{
	if (!m_currentRecommendation.valid) return 1;
	Real aggression = m_currentRecommendation.aggression;
	if (aggression < 0.2f) return 4;
	if (aggression < 0.4f) return 3;
	if (aggression < 0.6f) return 2;
	return 1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// DECISION METHODS - ML-influenced decision making
///////////////////////////////////////////////////////////////////////////////////////////////////

Bool AILearningPlayer::selectTeamToBuild()
{
	// If no ML recommendations yet, use parent logic
	if (!m_currentRecommendation.valid) {
		return AISkirmishPlayer::selectTeamToBuild();
	}

	// Find all buildable teams at highest priority (same as parent)
	Player::PlayerTeamList::const_iterator t;
	const Int invalidPri = -99999;
	Int hiPri = invalidPri;

	// Collect all teams that are possible to build
	Player::PlayerTeamList candidateList1;
	for (t = m_player->getPlayerTeams()->begin(); t != m_player->getPlayerTeams()->end(); ++t) {
		if (isAGoodIdeaToBuildTeam(*t)) {
			candidateList1.push_back(*t);
			Int pri = (*t)->getTemplateInfo()->m_productionPriority;
			if (pri > hiPri) {
				hiPri = pri;
			}
		}
	}

	// Try reinforcement first
	if (selectTeamToReinforce(hiPri)) {
		return true;
	}

	if (hiPri == invalidPri) {
		return false;
	}

	// Filter to highest priority teams and calculate ML weights
	Real totalWeight = 0.0f;
	struct WeightedTeam {
		TeamPrototype* proto;
		Real weight;
	};
	std::vector<WeightedTeam> weightedCandidates;
	weightedCandidates.reserve(16);

	for (t = candidateList1.begin(); t != candidateList1.end(); ++t) {
		if ((*t)->getTemplateInfo()->m_productionPriority == hiPri) {
			TeamCategory category = classifyTeam(*t);
			Real weight = getTeamCategoryWeight(category);

			// Minimum weight to ensure all teams have some chance
			weight = max(MLConfig::MIN_PRIORITY_WEIGHT, weight);

			WeightedTeam wt;
			wt.proto = *t;
			wt.weight = weight;
			weightedCandidates.push_back(wt);
			totalWeight += weight;

			ML_LOG(("ML Team Selection: %s category=%d weight=%.2f (inf=%.2f veh=%.2f air=%.2f)\n",
				(*t)->getName().str(), (int)category, weight,
				m_currentRecommendation.preferInfantry,
				m_currentRecommendation.preferVehicles,
				m_currentRecommendation.preferAircraft));
		}
	}

	if (weightedCandidates.empty() || totalWeight <= 0.0f) {
		return false;
	}

	// Weighted random selection
	Real randomValue = GameLogicRandomValue(0, 10000) / 10000.0f * totalWeight;
	Real cumulative = 0.0f;
	TeamPrototype* teamProto = NULL;

	for (size_t i = 0; i < weightedCandidates.size(); i++) {
		cumulative += weightedCandidates[i].weight;
		if (randomValue <= cumulative) {
			teamProto = weightedCandidates[i].proto;
			break;
		}
	}

	// Fallback to last team if rounding issues
	if (!teamProto && !weightedCandidates.empty()) {
		teamProto = weightedCandidates.back().proto;
	}

	if (teamProto) {
		ML_LOG(("ML Team Selected: %s\n", teamProto->getName().str()));

		buildSpecificAITeam(teamProto, false);
		m_readyToBuildTeam = false;
		m_teamTimer = m_teamSeconds * LOGICFRAMES_PER_SECOND;
		Money* money = m_player->getMoney();
		UnsignedInt currentMoney = money ? money->countMoney() : 0;
		if (currentMoney < TheAI->getAiData()->m_resourcesPoor) {
			m_teamTimer = m_teamTimer / TheAI->getAiData()->m_teamPoorMod;
		} else if (currentMoney > TheAI->getAiData()->m_resourcesWealthy) {
			m_teamTimer = m_teamTimer / TheAI->getAiData()->m_teamWealthyMod;
		}
		return true;
	}

	return false;
}

void AILearningPlayer::processBaseBuilding()
{
	// If no ML recommendations yet, use parent logic
	if (!m_currentRecommendation.valid) {
		AISkirmishPlayer::processBaseBuilding();
		return;
	}

	// Check if we're ready to build
	if (!m_readyToBuildStructure) {
		AISkirmishPlayer::processBaseBuilding();
		return;
	}

	// Iterate through build list to find candidate building
	// and apply ML priority weights
	Energy* energy = m_player->getEnergy();
	Bool hasSufficientPower = energy ? energy->hasSufficientPower() : true;
	const ThingTemplate* bestPlan = NULL;
	BuildListInfo* bestInfo = NULL;
	Real bestWeight = 0.0f;

	for (BuildListInfo* info = m_player->getBuildList(); info; info = info->getNext()) {
		AsciiString name = info->getTemplateName();
		if (name.isEmpty()) continue;

		const ThingTemplate* curPlan = TheThingFactory->findTemplate(name);
		if (!curPlan) continue;

		// Skip if already built or building
		Object* existingBldg = TheGameLogic->findObjectByID(info->getObjectID());
		if (existingBldg && !existingBldg->isEffectivelyDead()) continue;

		// Skip if already marked as priority (someone else queued it)
		if (info->isPriorityBuild()) continue;

		// Skip if on cooldown
		if (info->getObjectTimestamp() > TheGameLogic->getFrame()) continue;

		// Skip if we can't afford it or don't have prereqs
		if (!m_player->canBuild(curPlan)) continue;

		// Classify building and get weight
		BuildingCategory category = classifyBuilding(curPlan);
		Real weight = getBuildingCategoryWeight(category);

		// Power plants get priority boost when underpowered
		if (!hasSufficientPower && curPlan->isKindOf(KINDOF_FS_POWER)) {
			weight += 1.0f;
		}

		// Check if this should be delayed (low priority)
		if (shouldDelayBuilding(category)) {
			ML_LOG(("ML Building Delay: %s category=%d weight=%.2f\n",
				name.str(), (int)category, weight));
			continue;
		}

		// Track best candidate
		if (weight > bestWeight) {
			bestWeight = weight;
			bestPlan = curPlan;
			bestInfo = info;
		}
	}

	// If we found a good candidate, mark it as priority so parent builds it
	if (bestPlan && bestInfo) {
		ML_LOG(("ML Building Select: %s weight=%.2f (eco=%.2f def=%.2f mil=%.2f tech=%.2f)\n",
			bestInfo->getTemplateName().str(), bestWeight,
			m_currentRecommendation.priorityEconomy,
			m_currentRecommendation.priorityDefense,
			m_currentRecommendation.priorityMilitary,
			m_currentRecommendation.priorityTech));

		// Mark this building as priority - parent's processBaseBuilding will build it
		buildSpecificAIBuilding(bestInfo->getTemplateName());
	}

	// Call parent to handle actual construction (respects priority flag we just set)
	AISkirmishPlayer::processBaseBuilding();
}

void AILearningPlayer::processTeamBuilding()
{
	AISkirmishPlayer::processTeamBuilding();
}

void AILearningPlayer::checkReadyTeams()
{
	// If no ML recommendation, use parent logic
	if (!m_currentRecommendation.valid) {
		AISkirmishPlayer::checkReadyTeams();
		return;
	}

	// Calculate hold time based on aggression (linear relationship)
	// aggression=1.0 -> holdSeconds=0 (attack immediately)
	// aggression=0.0 -> holdSeconds=30 (max delay)
	Real holdSeconds = (1.0f - m_currentRecommendation.aggression) * MLConfig::MAX_ATTACK_HOLD_SECONDS;
	UnsignedInt holdFrames = (UnsignedInt)(holdSeconds * LOGICFRAMES_PER_SECOND);

	UnsignedInt currentFrame = TheGameLogic->getFrame();
	UnsignedInt lastAttack = m_lastAttackFrame;

	// Check if enough time has passed since last attack
	if (currentFrame < lastAttack + holdFrames) {
		// Count ready teams for logging
		Int readyTeams = 0;
		for (DLINK_ITERATOR<TeamInQueue> iter = iterate_TeamReadyQueue(); !iter.done(); iter.advance()) {
			readyTeams++;
		}
		if (readyTeams > 0) {
			m_teamsHeld = readyTeams;
			ML_LOG(("ML Attack Hold: aggr=%.2f, holding %d teams, wait %u more frames\n",
				m_currentRecommendation.aggression, readyTeams,
				(lastAttack + holdFrames) - currentFrame));
		}
		// Still holding - DON'T call parent
		return;
	}

	// Ready to attack - update last attack frame and proceed
	if (m_teamsHeld > 0) {
		ML_LOG(("ML Attack Release: aggr=%.2f, releasing %d teams\n",
			m_currentRecommendation.aggression, m_teamsHeld));
		m_teamsHeld = 0;
	}
	m_lastAttackFrame = currentFrame;

	AISkirmishPlayer::checkReadyTeams();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// HIERARCHICAL CONTROL - Tactical Layer
///////////////////////////////////////////////////////////////////////////////////////////////////

Bool AILearningPlayer::getBasePosition(Coord3D& outPos)
{
	// Return cached position if valid
	if (m_hasValidBasePos) {
		outPos = m_cachedBasePos;
		return true;
	}

	// Find command center
	for (Object* obj = TheGameLogic->getFirstObject(); obj; obj = obj->getNextObject()) {
		if (obj->getControllingPlayer() != m_player) continue;
		if (obj->isKindOf(KINDOF_COMMANDCENTER) && !obj->isEffectivelyDead()) {
			const Coord3D* pos = obj->getPosition();
			if (pos) {
				m_cachedBasePos = *pos;
				m_hasValidBasePos = true;
				outPos = m_cachedBasePos;
				return true;
			}
		}
	}

	return false;
}

Bool AILearningPlayer::teamNeedsTacticalUpdate(Team* team)
{
	if (!team) return false;

	TeamID teamId = team->getID();
	if (teamId >= MAX_TRACKED_TEAMS) return false;

	UnsignedInt lastFrame = m_lastTacticalFrame[teamId];
	UnsignedInt currentFrame = TheGameLogic->getFrame();

	return (currentFrame - lastFrame) >= m_tacticalDecisionInterval;
}

void AILearningPlayer::processTeamTactics()
{
	// Early exit if no batched mode or no recommendation
	if (!m_mlBridge.hasBatchedResponse()) return;

	const MLBatchedResponse& response = m_mlBridge.getLastBatchedResponse();

	// Process each team command in the response
	for (Int i = 0; i < response.numTeamCommands; i++) {
		const TacticalCommand& cmd = response.teamCommands[i];
		if (!cmd.valid) continue;

		// Find the team by ID
		Team* team = TheTeamFactory->findTeamByID((TeamID)cmd.teamId);
		if (!team) continue;

		// Verify this team belongs to our player
		if (team->getControllingPlayer() != m_player) continue;

		// Execute the command
		executeTacticalCommand(team, cmd);

		// Update tracking
		if (cmd.teamId < MAX_TRACKED_TEAMS) {
			m_lastTacticalFrame[cmd.teamId] = TheGameLogic->getFrame();
		}
	}
}

void AILearningPlayer::buildTeamTacticalState(Team* team, TacticalState& outState)
{
	// Use the shared state builder
	buildTacticalState(team, m_strategicOutput, m_player, outState);
}

void AILearningPlayer::executeTacticalCommand(Team* team, const TacticalCommand& cmd)
{
	if (!team) return;

	// Get team as AIGroup for issuing commands
	AIGroup* group = TheAI->createGroup();
	if (!group) return;

	team->getTeamAsAIGroup(group);

	// Calculate world position from normalized coordinates
	// Assuming map is approximately 3000x3000 units
	const Real mapScale = 3000.0f;
	Coord3D targetPos;
	targetPos.x = cmd.targetX * mapScale;
	targetPos.y = cmd.targetY * mapScale;
	targetPos.z = 0;

	ML_LOG(("TacticalCmd: team=%d action=%d pos=(%.1f,%.1f) attitude=%.2f\n",
		cmd.teamId, cmd.action, targetPos.x, targetPos.y, cmd.attitude));

	switch (cmd.action) {
		case TACTICAL_ATTACK_MOVE:
			group->groupAttackMoveToPosition(&targetPos, INT_MAX, CMD_FROM_AI);
			break;

		case TACTICAL_ATTACK_TARGET:
			// Find nearest enemy at target position to attack
			group->groupAttackMoveToPosition(&targetPos, INT_MAX, CMD_FROM_AI);
			break;

		case TACTICAL_DEFEND_POSITION:
			group->groupGuardPosition(&targetPos, GUARDMODE_NORMAL, CMD_FROM_AI);
			break;

		case TACTICAL_RETREAT: {
			Coord3D basePos;
			if (getBasePosition(basePos)) {
				group->groupMoveToPosition(&basePos, false, CMD_FROM_AI);
			}
			break;
		}

		case TACTICAL_HOLD:
			group->groupIdle(CMD_FROM_AI);
			break;

		case TACTICAL_HUNT:
			// Attack-move aggressively toward enemy
			group->groupAttackMoveToPosition(&targetPos, INT_MAX, CMD_FROM_AI);
			break;

		case TACTICAL_REINFORCE:
			// Move toward rally point
			group->groupMoveToPosition(&targetPos, false, CMD_FROM_AI);
			break;

		case TACTICAL_SPECIAL:
			// TODO: Implement special ability usage
			break;

		default:
			break;
	}

	// Cleanup group
	group->deleteInstance();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// HIERARCHICAL CONTROL - Micro Layer
///////////////////////////////////////////////////////////////////////////////////////////////////

Bool AILearningPlayer::unitNeedsMicroUpdate(Object* unit)
{
	if (!unit) return false;

	// Check if unit is in our tracking list
	ObjectID unitId = unit->getID();
	Int trackIdx = -1;

	for (Int i = 0; i < m_trackedUnitCount; i++) {
		if (m_trackedUnitIds[i] == unitId) {
			trackIdx = i;
			break;
		}
	}

	if (trackIdx < 0) {
		// Not tracked yet - add if space available and should be micro'd
		if (!shouldMicroUnit(unit)) return false;
		if (m_trackedUnitCount >= MAX_TRACKED_UNITS) return false;

		trackIdx = m_trackedUnitCount++;
		m_trackedUnitIds[trackIdx] = unitId;
		m_lastMicroFrame[trackIdx] = 0;
	}

	UnsignedInt lastFrame = m_lastMicroFrame[trackIdx];
	UnsignedInt currentFrame = TheGameLogic->getFrame();

	return (currentFrame - lastFrame) >= m_microDecisionInterval;
}

void AILearningPlayer::processMicroControl()
{
	// Early exit if no batched mode or no recommendation
	if (!m_mlBridge.hasBatchedResponse()) return;

	const MLBatchedResponse& response = m_mlBridge.getLastBatchedResponse();

	// Process each unit command in the response
	for (Int i = 0; i < response.numUnitCommands; i++) {
		const MicroCommand& cmd = response.unitCommands[i];
		if (!cmd.valid) continue;

		// Find the unit by ID
		Object* unit = TheGameLogic->findObjectByID(cmd.unitId);
		if (!unit || unit->isEffectivelyDead()) continue;

		// Verify this unit belongs to our player
		if (unit->getControllingPlayer() != m_player) continue;

		// Execute the command
		executeMicroCommand(unit, cmd);

		// Update tracking
		for (Int j = 0; j < m_trackedUnitCount; j++) {
			if (m_trackedUnitIds[j] == cmd.unitId) {
				m_lastMicroFrame[j] = TheGameLogic->getFrame();
				break;
			}
		}
	}

	// Cleanup stale tracked units
	Int writeIdx = 0;
	UnsignedInt currentFrame = TheGameLogic->getFrame();
	for (Int i = 0; i < m_trackedUnitCount; i++) {
		// Remove units not updated in 10 seconds
		if (currentFrame - m_lastMicroFrame[i] < 300) {
			if (writeIdx != i) {
				m_trackedUnitIds[writeIdx] = m_trackedUnitIds[i];
				m_lastMicroFrame[writeIdx] = m_lastMicroFrame[i];
			}
			writeIdx++;
		}
	}
	m_trackedUnitCount = writeIdx;
}

void AILearningPlayer::buildUnitMicroState(Object* unit, MicroState& outState)
{
	// Build team objective context (simplified - just pass direction to objective)
	Real teamObjective[4] = { 0, 0, 0, 0.5f };  // Default values

	// Try to get team objective from unit's team
	Team* team = unit->getTeam();
	if (team) {
		const Coord3D* teamPos = team->getEstimateTeamPosition();
		const Coord3D* unitPos = unit->getPosition();
		if (teamPos && unitPos) {
			// Direction to team center
			Real dx = teamPos->x - unitPos->x;
			Real dy = teamPos->y - unitPos->y;
			Real dist = sqrtf(dx * dx + dy * dy);
			if (dist > 0.001f) {
				teamObjective[1] = atan2f(dy, dx) / 3.14159265f;  // Normalized angle
			}
		}
	}

	buildMicroState(unit, teamObjective, outState);
}

void AILearningPlayer::executeMicroCommand(Object* unit, const MicroCommand& cmd)
{
	if (!unit) return;

	AIUpdateInterface* ai = unit->getAIUpdateInterface();
	if (!ai) return;

	const Coord3D* unitPos = unit->getPosition();
	if (!unitPos) return;

	ML_LOG(("MicroCmd: unit=%d action=%d angle=%.2f dist=%.2f\n",
		cmd.unitId, cmd.action, cmd.moveAngle, cmd.moveDistance));

	switch (cmd.action) {
		case MICRO_ATTACK_CURRENT:
			// Continue current target - no action needed
			break;

		case MICRO_ATTACK_NEAREST: {
			Real dist, angle;
			Object* target = findNearestEnemy(unit, &dist, &angle);
			if (target) {
				ai->aiAttackObject(target, false, CMD_FROM_AI);
			}
			break;
		}

		case MICRO_ATTACK_WEAKEST: {
			Object* target = findWeakestEnemy(unit, MicroConfig::COMBAT_DETECTION_RADIUS);
			if (target) {
				ai->aiAttackObject(target, false, CMD_FROM_AI);
			}
			break;
		}

		case MICRO_ATTACK_PRIORITY: {
			Object* target = findPriorityTarget(unit, MicroConfig::COMBAT_DETECTION_RADIUS);
			if (target) {
				ai->aiAttackObject(target, false, CMD_FROM_AI);
			}
			break;
		}

		case MICRO_MOVE_FORWARD: {
			// Move in specified direction
			Coord3D movePos;
			Real moveDist = cmd.moveDistance * 100.0f;  // Scale to game units
			movePos.x = unitPos->x + cosf(cmd.moveAngle) * moveDist;
			movePos.y = unitPos->y + sinf(cmd.moveAngle) * moveDist;
			movePos.z = unitPos->z;
			ai->aiMoveToPosition(&movePos, CMD_FROM_AI);
			break;
		}

		case MICRO_MOVE_BACKWARD: {
			// Kite - find nearest enemy and move away
			Real enemyDist, enemyAngle;
			Object* enemy = findNearestEnemy(unit, &enemyDist, &enemyAngle);
			if (enemy) {
				const Coord3D* enemyPos = enemy->getPosition();
				Coord3D kitePos;
				calculateKitePosition(unit, enemyPos, &kitePos);
				ai->aiMoveToPosition(&kitePos, CMD_FROM_AI);
			}
			break;
		}

		case MICRO_MOVE_FLANK: {
			// Circle strafe around nearest enemy
			Real enemyDist, enemyAngle;
			Object* enemy = findNearestEnemy(unit, &enemyDist, &enemyAngle);
			if (enemy) {
				const Coord3D* enemyPos = enemy->getPosition();
				Coord3D flankPos;
				calculateFlankPosition(unit, enemyPos, cmd.moveAngle > 0, &flankPos);
				ai->aiMoveToPosition(&flankPos, CMD_FROM_AI);
			}
			break;
		}

		case MICRO_HOLD_FIRE:
			ai->aiIdle(CMD_FROM_AI);
			break;

		case MICRO_USE_ABILITY:
			// TODO: Implement ability usage
			break;

		case MICRO_RETREAT: {
			Coord3D basePos;
			if (getBasePosition(basePos)) {
				ai->aiMoveToPosition(&basePos, CMD_FROM_AI);
			}
			break;
		}

		case MICRO_FOLLOW_TEAM:
			// Default team behavior - no explicit command needed
			break;

		default:
			break;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// GAME END DETECTION
///////////////////////////////////////////////////////////////////////////////////////////////////

// Helper: Check if a player has any surviving key units (CC or dozer)
static Bool playerHasKeyUnits(Player* player)
{
	for (Object* obj = TheGameLogic->getFirstObject(); obj; obj = obj->getNextObject()) {
		if (obj->getControllingPlayer() != player) continue;
		if (obj->isEffectivelyDead()) continue;
		if (obj->isKindOf(KINDOF_COMMANDCENTER) || obj->isKindOf(KINDOF_DOZER)) {
			return true;
		}
	}
	return false;
}

void AILearningPlayer::checkGameEnd()
{
	if (m_gameEndSent) return;
	if (!m_player) return;

	// Don't check for game end in the first 30 seconds (units still spawning)
	const UnsignedInt MIN_GAME_FRAMES = 30 * LOGICFRAMES_PER_SECOND;
	UnsignedInt currentFrame = TheGameLogic->getFrame();
	if (currentFrame < MIN_GAME_FRAMES) return;

	// FIX: Determine game end result first, then set flag before sending
	// This prevents potential re-entry issues
	Bool gameEnded = false;
	Bool victory = false;
	Real gameTime = (Real)currentFrame / (30.0f * 60.0f);
	Real armyStrength = 0.0f;
	const char* endReason = "";

	// Check if WE have lost (no CC and no dozer)
	if (!playerHasKeyUnits(m_player)) {
		gameEnded = true;
		victory = false;
		armyStrength = 0.0f;
		endReason = "Defeat: No CC or dozer";
	}
	else {
		// Check if ALL enemies are defeated (must check every enemy player)
		Int enemyCount = 0;
		Int aliveEnemyCount = 0;

		for (Int i = 0; i < MAX_PLAYER_COUNT; i++) {
			Player* otherPlayer = ThePlayerList->getNthPlayer(i);
			if (!otherPlayer || otherPlayer == m_player) continue;
			if (m_player->getRelationship(otherPlayer->getDefaultTeam()) != ENEMIES) continue;

			enemyCount++;
			if (playerHasKeyUnits(otherPlayer)) {
				aliveEnemyCount++;
			}
		}

		// Victory only if we found enemies AND all are defeated
		if (enemyCount > 0 && aliveEnemyCount == 0) {
			gameEnded = true;
			victory = true;
			armyStrength = calculateArmyStrength();
			endReason = "Victory: All enemies defeated";
		}
		// Game timeout handling (optional: end as draw after 60 minutes)
		else {
			const UnsignedInt MAX_GAME_FRAMES = 60 * 60 * LOGICFRAMES_PER_SECOND;  // 60 minutes
			if (currentFrame >= MAX_GAME_FRAMES) {
				gameEnded = true;
				armyStrength = calculateArmyStrength();
				victory = (armyStrength > 1.0f);
				endReason = "Timeout";
			}
		}
	}

	// Send game end if detected
	if (gameEnded) {
		// Set flag BEFORE sending to prevent any re-entry
		m_gameEndSent = true;
		m_mlBridge.sendGameEnd(victory, gameTime, armyStrength);
		ML_LOG(("%s at frame %d, army ratio %.2f\n", endReason, currentFrame, armyStrength));
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// SNAPSHOT METHODS
///////////////////////////////////////////////////////////////////////////////////////////////////

void AILearningPlayer::crc( Xfer *xfer )
{
	AISkirmishPlayer::crc(xfer);
}

void AILearningPlayer::xfer( Xfer *xfer )
{
	XferVersion currentVersion = 6;  // Bumped for m_lastAttackFrame type change to UnsignedInt
	XferVersion version = currentVersion;
	xfer->xferVersion( &version, currentVersion );

	AISkirmishPlayer::xfer(xfer);

	xfer->xferUnsignedInt(&m_frameCounter);
	xfer->xferReal(&m_lastFrameMoney);
	xfer->xferReal(&m_recentDamageTaken);
	xfer->xferUnsignedInt(&m_lastDamageFrame);
	xfer->xferInt(&m_teamsHeld);
	xfer->xferUnsignedInt(&m_lastAttackFrame);
	xfer->xferBool(&m_gameEndSent);
}

void AILearningPlayer::loadPostProcess()
{
	AISkirmishPlayer::loadPostProcess();
	m_mlBridge.connect();
}
