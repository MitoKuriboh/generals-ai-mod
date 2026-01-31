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
//  Learning AI Player - ML-driven strategic decisions                        //
//  Extends AISkirmishPlayer with machine learning capabilities               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

// AILearningPlayer.h
// ML-driven computerized opponent
// Author: Mito, 2025

#pragma once

#ifndef _AI_LEARNING_PLAYER_H_
#define _AI_LEARNING_PLAYER_H_

#include "Common/GameMemory.h"
#include "GameLogic/AISkirmishPlayer.h"
#include "GameLogic/MLBridge.h"

class BuildListInfo;
class SpecialPowerTemplate;
class TeamPrototype;

// Team type classification for ML-influenced selection
enum TeamCategory
{
	TEAM_CATEGORY_INFANTRY,
	TEAM_CATEGORY_VEHICLE,
	TEAM_CATEGORY_AIRCRAFT,
	TEAM_CATEGORY_MIXED,
	TEAM_CATEGORY_UNKNOWN
};

// Building type classification for ML-influenced priorities
enum BuildingCategory
{
	BUILDING_CATEGORY_ECONOMY,    // Supply, refineries, etc.
	BUILDING_CATEGORY_POWER,      // Power plants
	BUILDING_CATEGORY_DEFENSE,    // Turrets, walls
	BUILDING_CATEGORY_MILITARY,   // Barracks, war factory, airfield
	BUILDING_CATEGORY_TECH,       // Tech center, strategy center
	BUILDING_CATEGORY_SUPER,      // Superweapons
	BUILDING_CATEGORY_UNKNOWN
};

/**
 * ML-driven computer-controlled opponent.
 * Extends AISkirmishPlayer to use machine learning for strategic decisions
 * while leveraging the existing infrastructure for command execution.
 */
class AILearningPlayer : public AISkirmishPlayer
{
	MEMORY_POOL_GLUE_WITH_USERLOOKUP_CREATE( AILearningPlayer, "AILearningPlayer" )

public:
	AILearningPlayer( Player *p );
	// Note: destructor declared by MEMORY_POOL_GLUE macro

	// AIPlayer interface methods
	virtual void update();
	virtual void newMap();
	virtual void onUnitProduced( Object *factory, Object *unit );

	// Identify as learning AI
	virtual Bool isLearningAI(void) { return true; }

protected:
	// Snapshot methods
	virtual void crc( Xfer *xfer );
	virtual void xfer( Xfer *xfer );
	virtual void loadPostProcess( void );

	// Decision methods - override for ML
	virtual Bool selectTeamToBuild( void );
	virtual void processBaseBuilding( void );
	virtual void processTeamBuilding( void );

	// Override to control attack timing based on aggression
	virtual void checkReadyTeams( void );

private:
	// ML Bridge communication
	void exportStateToML();
	void buildGameState(MLGameState& state);
	void processMLRecommendations();

	// Helper methods for state extraction
	void countForces(Real* infantry, Real* vehicles, Real* aircraft, Real* structures, Bool own);
	Real calculateTechLevel();
	Real calculateBaseThreat();
	Real calculateArmyStrength();
	Real calculateDistanceToEnemy();

	// Team classification helpers
	TeamCategory classifyTeam(TeamPrototype* proto);
	Real getTeamCategoryWeight(TeamCategory category);

	// Building classification helpers
	BuildingCategory classifyBuilding(const ThingTemplate* tmpl);
	Real getBuildingCategoryWeight(BuildingCategory category);
	Bool shouldDelayBuilding(BuildingCategory category);

	// Attack timing based on aggression
	Bool shouldHoldTeam();
	Int getMinArmySizeForAttack();

	// Check for game end and notify ML
	void checkGameEnd();

	// ML Bridge instance
	MLBridge m_mlBridge;

	// Current ML recommendation
	MLRecommendation m_currentRecommendation;

	// State tracking
	UnsignedInt m_frameCounter;

	// Attack timing state
	Int m_teamsHeld;           // Number of teams waiting to attack
	Int m_lastAttackFrame;     // Frame when last attack was launched

	// Game end tracking
	Bool m_gameEndSent;        // True if we've sent game end notification

	// ML decision interval (30 frames = ~1 second at 30 FPS)
	static const UnsignedInt ML_DECISION_INTERVAL = 30;
};

#endif // _AI_LEARNING_PLAYER_H_
