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
#include "Common/GameMemory.h"
#include "Common/GlobalData.h"
#include "Common/Player.h"
#include "Common/PlayerList.h"
#include "Common/Xfer.h"
#include "Common/ThingTemplate.h"
#include "Common/ThingFactory.h"
#include "Common/Team.h"
#include "GameLogic/GameLogic.h"
#include "GameLogic/Object.h"
#include "GameLogic/AILearningPlayer.h"
#include "GameLogic/AI.h"

// Enable ML state logging
#define DEBUG_ML_STATE 1

#ifdef DEBUG_ML_STATE
#define ML_LOG(x) DEBUG_LOG(x)
#else
#define ML_LOG(x)
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// CONSTRUCTOR / DESTRUCTOR
///////////////////////////////////////////////////////////////////////////////////////////////////

AILearningPlayer::AILearningPlayer( Player *p ) : AISkirmishPlayer(p),
	m_frameCounter(0),
	m_teamsHeld(0),
	m_lastAttackFrame(0),
	m_gameEndSent(false)
{
	m_currentRecommendation.clear();
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
	m_frameCounter++;

	// Check for game end
	checkGameEnd();

	// Every 30 frames, communicate with ML bridge
	if (m_frameCounter % ML_DECISION_INTERVAL == 0) {
		exportStateToML();
		processMLRecommendations();
	}

	// Run normal skirmish AI
	AISkirmishPlayer::update();
}

void AILearningPlayer::newMap()
{
	m_frameCounter = 0;
	m_teamsHeld = 0;
	m_lastAttackFrame = 0;
	m_gameEndSent = false;
	m_currentRecommendation.clear();

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
	UnsignedInt money = m_player->getMoney()->countMoney();
	state.money = (Real)log10((double)(money + 1));

	// Power
	Energy* energy = m_player->getEnergy();
	if (energy) {
		state.powerBalance = (Real)(energy->getProduction() - energy->getConsumption());
	}

	// Count forces (simplified)
	countForces(state.ownInfantry, state.ownVehicles, state.ownAircraft, state.ownStructures, true);
	countForces(state.enemyInfantry, state.enemyVehicles, state.enemyAircraft, state.enemyStructures, false);

	// Time and metrics
	state.gameTimeMinutes = (Real)TheGameLogic->getFrame() / (30.0f * 60.0f);
	state.techLevel = calculateTechLevel();
	state.baseThreat = calculateBaseThreat();
	state.armyStrength = calculateArmyStrength();
	state.distanceToEnemy = calculateDistanceToEnemy();
	state.underAttack = 0.0f;
}

void AILearningPlayer::countForces(Real* infantry, Real* vehicles, Real* aircraft, Real* structures, Bool own)
{
	infantry[0] = infantry[1] = infantry[2] = 0.0f;
	vehicles[0] = vehicles[1] = vehicles[2] = 0.0f;
	aircraft[0] = aircraft[1] = aircraft[2] = 0.0f;
	structures[0] = structures[1] = structures[2] = 0.0f;

	Int infantryCount = 0, vehicleCount = 0, aircraftCount = 0, structureCount = 0;

	for (Object* obj = TheGameLogic->getFirstObject(); obj; obj = obj->getNextObject()) {
		if (obj->isEffectivelyDead()) continue;

		Bool isOwn = (obj->getControllingPlayer() == m_player);
		if (isOwn != own) continue;

		if (obj->isKindOf(KINDOF_INFANTRY)) {
			infantryCount++;
		} else if (obj->isKindOf(KINDOF_VEHICLE)) {
			vehicleCount++;
		} else if (obj->isKindOf(KINDOF_AIRCRAFT)) {
			aircraftCount++;
		} else if (obj->isKindOf(KINDOF_STRUCTURE)) {
			structureCount++;
		}
	}

	infantry[0] = (Real)log10((double)(infantryCount + 1));
	vehicles[0] = (Real)log10((double)(vehicleCount + 1));
	aircraft[0] = (Real)log10((double)(aircraftCount + 1));
	structures[0] = (Real)log10((double)(structureCount + 1));
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
			basePos = *obj->getPosition();
			hasBase = true;
			break;
		}
	}

	if (!hasBase) return 1.0f;

	Int nearbyEnemies = 0;
	const Real threatRadius = 500.0f;

	for (Object* obj = TheGameLogic->getFirstObject(); obj; obj = obj->getNextObject()) {
		if (obj->getControllingPlayer() == m_player) continue;
		if (obj->isEffectivelyDead()) continue;
		if (!obj->isKindOf(KINDOF_CAN_ATTACK)) continue;

		const Coord3D* pos = obj->getPosition();
		Real dx = pos->x - basePos.x;
		Real dy = pos->y - basePos.y;
		Real dist = sqrt(dx*dx + dy*dy);

		if (dist < threatRadius) {
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

	if (enemyUnits < 1) return 2.0f;
	return min(2.0f, (Real)ownUnits / (Real)enemyUnits);
}

Real AILearningPlayer::calculateDistanceToEnemy()
{
	Coord3D myBase = {0, 0, 0};
	Coord3D enemyBase = {0, 0, 0};
	Bool foundMy = false, foundEnemy = false;

	for (Object* obj = TheGameLogic->getFirstObject(); obj; obj = obj->getNextObject()) {
		if (!obj->isKindOf(KINDOF_COMMANDCENTER)) continue;
		if (obj->isEffectivelyDead()) continue;

		if (obj->getControllingPlayer() == m_player) {
			myBase = *obj->getPosition();
			foundMy = true;
		} else {
			Player* otherPlayer = obj->getControllingPlayer();
			if (otherPlayer && m_player->getRelationship(otherPlayer->getDefaultTeam()) == ENEMIES) {
				enemyBase = *obj->getPosition();
				foundEnemy = true;
			}
		}
	}

	if (!foundMy || !foundEnemy) return 1.0f;

	Real dx = enemyBase.x - myBase.x;
	Real dy = enemyBase.y - myBase.y;
	return min(1.0f, sqrt(dx*dx + dy*dy) / 3000.0f);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// TEAM CLASSIFICATION - Simplified
///////////////////////////////////////////////////////////////////////////////////////////////////

TeamCategory AILearningPlayer::classifyTeam(TeamPrototype* proto)
{
	// Simplified: just return mixed for now
	// Full implementation needs proper team template iteration
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
	Real weight = getBuildingCategoryWeight(category);
	return (weight < 0.15f);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// ATTACK TIMING
///////////////////////////////////////////////////////////////////////////////////////////////////

Bool AILearningPlayer::shouldHoldTeam()
{
	if (!m_currentRecommendation.valid) return false;
	return (m_currentRecommendation.aggression < 0.3f);
}

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
// DECISION METHODS - For now, just use parent logic
///////////////////////////////////////////////////////////////////////////////////////////////////

Bool AILearningPlayer::selectTeamToBuild()
{
	return AISkirmishPlayer::selectTeamToBuild();
}

void AILearningPlayer::processBaseBuilding()
{
	AISkirmishPlayer::processBaseBuilding();
}

void AILearningPlayer::processTeamBuilding()
{
	AISkirmishPlayer::processTeamBuilding();
}

void AILearningPlayer::checkReadyTeams()
{
	AISkirmishPlayer::checkReadyTeams();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// GAME END DETECTION
///////////////////////////////////////////////////////////////////////////////////////////////////

void AILearningPlayer::checkGameEnd()
{
	if (m_gameEndSent) return;
	if (!m_player) return;

	Bool hasCommandCenter = false;
	Bool hasDozer = false;

	for (Object* obj = TheGameLogic->getFirstObject(); obj; obj = obj->getNextObject()) {
		if (obj->getControllingPlayer() != m_player) continue;
		if (obj->isEffectivelyDead()) continue;

		if (obj->isKindOf(KINDOF_COMMANDCENTER)) hasCommandCenter = true;
		if (obj->isKindOf(KINDOF_DOZER)) hasDozer = true;
	}

	// We've lost if we have no command center and no dozer
	if (!hasCommandCenter && !hasDozer) {
		Real gameTime = (Real)TheGameLogic->getFrame() / (30.0f * 60.0f);
		m_mlBridge.sendGameEnd(false, gameTime, 0.0f);
		m_gameEndSent = true;
		DEBUG_LOG(("AILearningPlayer: Defeat detected\n"));
		return;
	}

	// Check if all enemies defeated
	Bool anyEnemyAlive = false;
	for (Int i = 0; i < MAX_PLAYER_COUNT; i++) {
		Player* otherPlayer = ThePlayerList->getNthPlayer(i);
		if (!otherPlayer || otherPlayer == m_player) continue;
		if (m_player->getRelationship(otherPlayer->getDefaultTeam()) != ENEMIES) continue;

		for (Object* obj = TheGameLogic->getFirstObject(); obj; obj = obj->getNextObject()) {
			if (obj->getControllingPlayer() != otherPlayer) continue;
			if (obj->isEffectivelyDead()) continue;
			if (obj->isKindOf(KINDOF_COMMANDCENTER) || obj->isKindOf(KINDOF_DOZER)) {
				anyEnemyAlive = true;
				break;
			}
		}
		if (anyEnemyAlive) break;
	}

	if (!anyEnemyAlive) {
		Real gameTime = (Real)TheGameLogic->getFrame() / (30.0f * 60.0f);
		m_mlBridge.sendGameEnd(true, gameTime, calculateArmyStrength());
		m_gameEndSent = true;
		DEBUG_LOG(("AILearningPlayer: Victory detected\n"));
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
	XferVersion currentVersion = 4;
	XferVersion version = currentVersion;
	xfer->xferVersion( &version, currentVersion );

	AISkirmishPlayer::xfer(xfer);

	xfer->xferUnsignedInt(&m_frameCounter);
	xfer->xferInt(&m_teamsHeld);
	xfer->xferInt(&m_lastAttackFrame);
	xfer->xferBool(&m_gameEndSent);
}

void AILearningPlayer::loadPostProcess()
{
	AISkirmishPlayer::loadPostProcess();
	m_mlBridge.connect();
}
