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
//  ML Bridge - Communication interface between game and Python ML system     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

// MLBridge.cpp
// Named pipe communication with Python ML process
// Author: Mito, 2025

#include "PreRTS.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <math.h>   // For isfinite()
#include <float.h>  // For _finite() fallback on MSVC

#include "GameLogic/MLBridge.h"
#include "GameLogic/TacticalState.h"
#include "GameLogic/MicroState.h"
#include "GameLogic/GameLogic.h"

// MSVC doesn't have isfinite in older versions, use _finite instead
#ifndef isfinite
#define isfinite(x) _finite(x)
#endif

#ifdef _INTERNAL
// for occasional debugging...
//#pragma optimize("", off)
//#pragma MESSAGE("************************************** WARNING, optimization disabled for debugging purposes")
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// MLGameState
///////////////////////////////////////////////////////////////////////////////////////////////////

void MLGameState::clear()
{
	playerIndex = 0;
	money = 0.0f;
	powerBalance = 0.0f;
	incomeRate = 0.0f;
	supplyUsed = 0.0f;

	for (int i = 0; i < 3; i++) {
		ownInfantry[i] = 0.0f;
		ownVehicles[i] = 0.0f;
		ownAircraft[i] = 0.0f;
		ownStructures[i] = 0.0f;
		enemyInfantry[i] = 0.0f;
		enemyVehicles[i] = 0.0f;
		enemyAircraft[i] = 0.0f;
		enemyStructures[i] = 0.0f;
	}

	gameTimeMinutes = 0.0f;
	techLevel = 0.0f;
	baseThreat = 0.0f;
	armyStrength = 0.0f;
	underAttack = 0.0f;
	distanceToEnemy = 0.0f;

	isUSA = 0.0f;
	isChina = 0.0f;
	isGLA = 0.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// MLRecommendation
///////////////////////////////////////////////////////////////////////////////////////////////////

void MLRecommendation::clear()
{
	// Default to balanced priorities
	priorityEconomy = 0.25f;
	priorityDefense = 0.25f;
	priorityMilitary = 0.25f;
	priorityTech = 0.25f;

	// Default to balanced army
	preferInfantry = 0.33f;
	preferVehicles = 0.34f;
	preferAircraft = 0.33f;

	// Moderate aggression
	aggression = 0.5f;

	// Auto-select target
	targetPlayer = -1;

	valid = false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// MLBridge Constructor / Destructor
///////////////////////////////////////////////////////////////////////////////////////////////////

MLBridge::MLBridge() :
	m_pipeHandle(INVALID_HANDLE_VALUE),
	m_connected(false),
	m_hasRecommendation(false),
	m_lastConnectAttempt(0),
	m_lastRecommendationFrame(0),
	m_trainerLaunched(false),
	m_trainerProcess(NULL),
	m_batchedModeEnabled(false),
	m_hasBatchedResponse(false)
{
	m_lastRecommendation.clear();
	m_lastBatchedResponse.clear();
	memset(m_readBuffer, 0, READ_BUFFER_SIZE);
	memset(m_batchedReadBuffer, 0, BATCHED_BUFFER_SIZE);
}

MLBridge::~MLBridge()
{
	disconnect();

	// Clean up trainer process handle (don't terminate - let it finish)
	if (m_trainerProcess != NULL) {
		CloseHandle((HANDLE)m_trainerProcess);
		m_trainerProcess = NULL;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Connection Management
///////////////////////////////////////////////////////////////////////////////////////////////////

Bool MLBridge::connect()
{
	if (m_connected) {
		return true;
	}

	// Throttle reconnection attempts
	UnsignedInt currentFrame = TheGameLogic ? TheGameLogic->getFrame() : 0;
	if (currentFrame - m_lastConnectAttempt < RECONNECT_INTERVAL_FRAMES) {
		return false;
	}
	m_lastConnectAttempt = currentFrame;

	// Try to connect to existing pipe (Python server should create it)
	HANDLE pipe = CreateFileA(
		getPipeName(),
		GENERIC_READ | GENERIC_WRITE,
		0,              // No sharing
		NULL,           // Default security
		OPEN_EXISTING,  // Must already exist
		FILE_FLAG_OVERLAPPED,  // For non-blocking I/O
		NULL            // No template
	);

	if (pipe == INVALID_HANDLE_VALUE) {
		DWORD error = GetLastError();
		if (error == ERROR_FILE_NOT_FOUND) {
			// Pipe doesn't exist - try to launch the trainer
			if (!m_trainerLaunched) {
				DEBUG_LOG(("MLBridge: Pipe not found, launching trainer...\n"));
				launchTrainer();
			}
		} else if (error != ERROR_PIPE_BUSY) {
			DEBUG_LOG(("MLBridge: Failed to connect, error %d\n", error));
		}
		return false;
	}

	// Set pipe to message mode
	DWORD mode = PIPE_READMODE_BYTE;
	if (!SetNamedPipeHandleState(pipe, &mode, NULL, NULL)) {
		DEBUG_LOG(("MLBridge: Failed to set pipe mode\n"));
		CloseHandle(pipe);
		return false;
	}

	m_pipeHandle = pipe;
	m_connected = true;
	DEBUG_LOG(("MLBridge: Connected to Python ML server\n"));

	return true;
}

void MLBridge::disconnect()
{
	if (m_pipeHandle != INVALID_HANDLE_VALUE) {
		CloseHandle((HANDLE)m_pipeHandle);
		m_pipeHandle = INVALID_HANDLE_VALUE;
	}
	m_connected = false;
	m_hasRecommendation = false;
	// FIX I1: Reset recommendation frame on disconnect to allow immediate reconnect
	m_lastRecommendationFrame = 0;
	DEBUG_LOG(("MLBridge: Disconnected\n"));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Trainer Process Launch
///////////////////////////////////////////////////////////////////////////////////////////////////

// FIX: Helper to check if a file exists
static Bool fileExists(const char* path)
{
	DWORD attrs = GetFileAttributesA(path);
	return attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY);
}

// FIX: Helper to find Python executable in common locations
static Bool findPythonExecutable(char* outPath, size_t outPathSize)
{
	// Try 'where python' to find Python in PATH
	FILE* pipe = _popen("where python 2>nul", "r");
	if (pipe) {
		if (fgets(outPath, (int)outPathSize - 1, pipe)) {
			// Remove newline
			char* newline = strchr(outPath, '\n');
			if (newline) *newline = '\0';
			newline = strchr(outPath, '\r');
			if (newline) *newline = '\0';

			_pclose(pipe);
			if (fileExists(outPath)) {
				return true;
			}
		} else {
			_pclose(pipe);
		}
	}

	// Try common Python installation paths
	const char* commonPaths[] = {
		"C:\\Python39\\python.exe",
		"C:\\Python310\\python.exe",
		"C:\\Python311\\python.exe",
		"C:\\Python312\\python.exe",
		"C:\\Program Files\\Python39\\python.exe",
		"C:\\Program Files\\Python310\\python.exe",
		"C:\\Program Files\\Python311\\python.exe",
		"C:\\Program Files\\Python312\\python.exe",
		"C:\\Users\\Public\\python\\python.exe",
		NULL
	};

	for (int i = 0; commonPaths[i]; i++) {
		if (fileExists(commonPaths[i])) {
			strncpy(outPath, commonPaths[i], outPathSize - 1);
			outPath[outPathSize - 1] = '\0';
			return true;
		}
	}

	// Last resort: just use python.exe and hope it's in PATH
	strncpy(outPath, "python.exe", outPathSize - 1);
	outPath[outPathSize - 1] = '\0';
	return true;  // Will fail later if not found
}

Bool MLBridge::launchTrainer()
{
	if (m_trainerLaunched) {
		return true;  // Already launched
	}

	m_trainerLaunched = true;  // Mark as attempted even if it fails

	// FIX: Check GENERALS_AI_DIR environment variable first
	char envDir[MAX_PATH];
	DWORD envLen = GetEnvironmentVariableA("GENERALS_AI_DIR", envDir, MAX_PATH);

	char scriptPath[MAX_PATH];
	char workDir[MAX_PATH];
	Bool found = false;

	// Search order:
	// 1. GENERALS_AI_DIR environment variable
	// 2. Game directory / python
	// 3. C:\Users\Public\generals-ai-mod\python (common install location)
	// 4. C:\Users\Public\game-ai-agent\python (alternate location)

	const char* searchDirs[5];
	int numDirs = 0;

	if (envLen > 0 && envLen < MAX_PATH) {
		searchDirs[numDirs++] = envDir;
	}

	// Get the game's directory
	char gameDir[MAX_PATH];
	GetModuleFileNameA(NULL, gameDir, MAX_PATH);
	char* lastSlash = strrchr(gameDir, '\\');
	if (lastSlash) {
		*lastSlash = '\0';
	}
	searchDirs[numDirs++] = gameDir;

	// Common installation locations
	searchDirs[numDirs++] = "C:\\Users\\Public\\generals-ai-mod";
	searchDirs[numDirs++] = "C:\\Users\\Public\\game-ai-agent";
	searchDirs[numDirs] = NULL;

	// Search for the script
	for (int i = 0; i < numDirs && !found; i++) {
		int written = snprintf(scriptPath, sizeof(scriptPath),
			"%s\\python\\train_manual.py", searchDirs[i]);
		if (written > 0 && written < (int)sizeof(scriptPath)) {
			if (fileExists(scriptPath)) {
				snprintf(workDir, sizeof(workDir), "%s\\python", searchDirs[i]);
				found = true;
				DEBUG_LOG(("MLBridge: Found trainer at %s\n", scriptPath));
			}
		}
	}

	if (!found) {
		DEBUG_LOG(("MLBridge: Trainer script not found. Searched:\n"));
		for (int i = 0; i < numDirs; i++) {
			DEBUG_LOG(("  - %s\\python\\train_manual.py\n", searchDirs[i]));
		}
		DEBUG_LOG(("Set GENERALS_AI_DIR environment variable to fix.\n"));
		return false;
	}

	// FIX: Find Python executable
	char pythonPath[MAX_PATH];
	if (!findPythonExecutable(pythonPath, sizeof(pythonPath))) {
		DEBUG_LOG(("MLBridge: Python not found. Install Python and add to PATH.\n"));
		return false;
	}

	// Build command line - try pythonw first (no console), then python
	char cmdLine[MAX_PATH * 2];
	Bool usePythonW = false;

	// Check if pythonw exists in same directory as python
	char pythonwPath[MAX_PATH];
	strncpy(pythonwPath, pythonPath, sizeof(pythonwPath) - 1);
	char* pythonExe = strstr(pythonwPath, "python.exe");
	if (pythonExe) {
		// Replace python.exe with pythonw.exe
		strncpy(pythonExe, "pythonw.exe", strlen("pythonw.exe") + 1);
		if (fileExists(pythonwPath)) {
			usePythonW = true;
		}
	}

	snprintf(cmdLine, sizeof(cmdLine), "\"%s\" \"%s\" --episodes 9999",
		usePythonW ? pythonwPath : pythonPath, scriptPath);

	DEBUG_LOG(("MLBridge: Launching trainer: %s\n", cmdLine));
	DEBUG_LOG(("MLBridge: Working directory: %s\n", workDir));

	// Set up process creation
	STARTUPINFOA si;
	PROCESS_INFORMATION pi;
	ZeroMemory(&si, sizeof(si));
	si.cb = sizeof(si);
	ZeroMemory(&pi, sizeof(pi));

	// Create the process
	BOOL success = CreateProcessA(
		NULL,           // Application name (NULL = use command line)
		cmdLine,        // Command line
		NULL,           // Process security attributes
		NULL,           // Thread security attributes
		FALSE,          // Don't inherit handles
		CREATE_NO_WINDOW | DETACHED_PROCESS,  // Creation flags
		NULL,           // Use parent's environment
		workDir,        // Working directory
		&si,            // Startup info
		&pi             // Process info output
	);

	if (!success) {
		DWORD error = GetLastError();
		(void)error;  // Suppress unused variable warning in release builds
		DEBUG_LOG(("MLBridge: Failed to launch trainer, error %d\n", error));

		// Try with python.exe if we used pythonw
		if (usePythonW) {
			snprintf(cmdLine, sizeof(cmdLine), "\"%s\" \"%s\" --episodes 9999",
				pythonPath, scriptPath);
			DEBUG_LOG(("MLBridge: Retrying with python.exe: %s\n", cmdLine));

			success = CreateProcessA(NULL, cmdLine, NULL, NULL, FALSE,
				CREATE_NO_WINDOW | DETACHED_PROCESS, NULL, workDir, &si, &pi);
		}

		if (!success) {
			DEBUG_LOG(("MLBridge: All launch attempts failed, error %d\n", GetLastError()));
			DEBUG_LOG(("MLBridge: Ensure Python is installed and in PATH.\n"));
			// FIX B10: Reset launch flag on complete failure to allow retry
			m_trainerLaunched = false;
			return false;
		}
	}

	// Store process handle for cleanup, close thread handle
	m_trainerProcess = pi.hProcess;
	CloseHandle(pi.hThread);

	DEBUG_LOG(("MLBridge: Trainer launched (PID %d)\n", pi.dwProcessId));

	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Message I/O
///////////////////////////////////////////////////////////////////////////////////////////////////

Bool MLBridge::writeMessage(const char* data, UnsignedInt length)
{
	if (!m_connected || m_pipeHandle == INVALID_HANDLE_VALUE) {
		return false;
	}

	// FIX B1: Atomic write - combine length prefix and data into single buffer
	// This prevents corruption if interrupted between writes
	static const UnsignedInt MAX_ATOMIC_BUFFER = 65536;  // 64KB limit
	UnsignedInt totalLength = sizeof(UnsignedInt) + length;

	if (totalLength > MAX_ATOMIC_BUFFER) {
		DEBUG_LOG(("MLBridge: Message too large for atomic write (%d bytes)\n", totalLength));
		return false;
	}

	// Build combined buffer: [4-byte length][data]
	char atomicBuffer[MAX_ATOMIC_BUFFER];
	memcpy(atomicBuffer, &length, sizeof(UnsignedInt));
	memcpy(atomicBuffer + sizeof(UnsignedInt), data, length);

	// Single atomic write
	DWORD bytesWritten = 0;
	if (!WriteFile((HANDLE)m_pipeHandle, atomicBuffer, totalLength, &bytesWritten, NULL)) {
		DEBUG_LOG(("MLBridge: Atomic write failed, disconnecting\n"));
		disconnect();
		return false;
	}

	if (bytesWritten != totalLength) {
		DEBUG_LOG(("MLBridge: Partial write (%d/%d bytes), disconnecting\n", bytesWritten, totalLength));
		disconnect();
		return false;
	}

	return true;
}

Bool MLBridge::readMessage(char* buffer, UnsignedInt bufferSize, UnsignedInt& outLength)
{
	if (!m_connected || m_pipeHandle == INVALID_HANDLE_VALUE) {
		return false;
	}

	// Check if data is available (non-blocking)
	DWORD bytesAvailable = 0;
	if (!PeekNamedPipe((HANDLE)m_pipeHandle, NULL, 0, NULL, &bytesAvailable, NULL)) {
		DWORD error = GetLastError();
		if (error == ERROR_BROKEN_PIPE || error == ERROR_PIPE_NOT_CONNECTED) {
			DEBUG_LOG(("MLBridge: Pipe broken, disconnecting\n"));
			disconnect();
		}
		return false;
	}

	if (bytesAvailable < sizeof(UnsignedInt)) {
		return false;  // No complete message yet
	}

	// Read length prefix
	DWORD bytesRead = 0;
	UnsignedInt msgLength = 0;
	if (!ReadFile((HANDLE)m_pipeHandle, &msgLength, sizeof(UnsignedInt), &bytesRead, NULL)) {
		DEBUG_LOG(("MLBridge: Read length failed\n"));
		disconnect();
		return false;
	}

	// FIX: Leave room for null terminator to prevent buffer overflow
	if (msgLength >= bufferSize - 1) {
		DEBUG_LOG(("MLBridge: Message too large (%d bytes, max %d)\n", msgLength, bufferSize - 1));
		disconnect();
		return false;
	}

	// Read message data
	if (!ReadFile((HANDLE)m_pipeHandle, buffer, msgLength, &bytesRead, NULL)) {
		DEBUG_LOG(("MLBridge: Read data failed\n"));
		disconnect();
		return false;
	}

	buffer[msgLength] = '\0';
	outLength = msgLength;
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// State Export
///////////////////////////////////////////////////////////////////////////////////////////////////

// Protocol version - increment when state format changes
static const Int ML_PROTOCOL_VERSION = 2;

AsciiString MLBridge::stateToJson(const MLGameState& state)
{
	// Simple JSON serialization (no external library needed)
	// FIX: Use snprintf with buffer size tracking to prevent overflow
	static const size_t BUFFER_SIZE = 2048;
	char buffer[BUFFER_SIZE];
	size_t remaining = BUFFER_SIZE;
	char* pos = buffer;
	int written;

	written = snprintf(pos, remaining,
		"{"
		"\"version\":%d,"
		"\"player\":%d,",
		ML_PROTOCOL_VERSION,
		state.playerIndex);

	if (written < 0 || (size_t)written >= remaining) {
		DEBUG_LOG(("MLBridge: stateToJson buffer overflow in header\n"));
		return AsciiString("{}");
	}
	pos += written;
	remaining -= written;

	// Continue with remaining fields
	written = snprintf(pos, remaining,
		"\"money\":%.2f,"
		"\"power\":%.2f,"
		"\"income\":%.2f,"
		"\"supply\":%.2f,"
		"\"own_infantry\":[%.2f,%.2f,%.2f],"
		"\"own_vehicles\":[%.2f,%.2f,%.2f],"
		"\"own_aircraft\":[%.2f,%.2f,%.2f],"
		"\"own_structures\":[%.2f,%.2f,%.2f],"
		"\"enemy_infantry\":[%.2f,%.2f,%.2f],"
		"\"enemy_vehicles\":[%.2f,%.2f,%.2f],"
		"\"enemy_aircraft\":[%.2f,%.2f,%.2f],"
		"\"enemy_structures\":[%.2f,%.2f,%.2f],"
		"\"game_time\":%.2f,"
		"\"tech_level\":%.2f,"
		"\"base_threat\":%.2f,"
		"\"army_strength\":%.2f,"
		"\"under_attack\":%.1f,"
		"\"distance_to_enemy\":%.2f,"
		"\"is_usa\":%.1f,"
		"\"is_china\":%.1f,"
		"\"is_gla\":%.1f"
		"}",
		state.money,
		state.powerBalance,
		state.incomeRate,
		state.supplyUsed,
		state.ownInfantry[0], state.ownInfantry[1], state.ownInfantry[2],
		state.ownVehicles[0], state.ownVehicles[1], state.ownVehicles[2],
		state.ownAircraft[0], state.ownAircraft[1], state.ownAircraft[2],
		state.ownStructures[0], state.ownStructures[1], state.ownStructures[2],
		state.enemyInfantry[0], state.enemyInfantry[1], state.enemyInfantry[2],
		state.enemyVehicles[0], state.enemyVehicles[1], state.enemyVehicles[2],
		state.enemyAircraft[0], state.enemyAircraft[1], state.enemyAircraft[2],
		state.enemyStructures[0], state.enemyStructures[1], state.enemyStructures[2],
		state.gameTimeMinutes,
		state.techLevel,
		state.baseThreat,
		state.armyStrength,
		state.underAttack,
		state.distanceToEnemy,
		state.isUSA,
		state.isChina,
		state.isGLA
	);

	if (written < 0 || (size_t)written >= remaining) {
		DEBUG_LOG(("MLBridge: stateToJson buffer overflow in body\n"));
		return AsciiString("{}");
	}

	return AsciiString(buffer);
}

// FIX R1: Validate state values before sending to ML server
// Returns true if all values are finite (not NaN or Inf)
static Bool validateStateValue(Real value, const char* name)
{
	if (!isfinite(value)) {
		DEBUG_LOG(("MLBridge: Invalid state value for '%s': %f\n", name, value));
		return false;
	}
	return true;
}

Bool MLBridge::validateState(const MLGameState& state)
{
	Bool valid = true;

	// Validate all float fields
	valid &= validateStateValue(state.money, "money");
	valid &= validateStateValue(state.powerBalance, "powerBalance");
	valid &= validateStateValue(state.incomeRate, "incomeRate");
	valid &= validateStateValue(state.supplyUsed, "supplyUsed");

	for (int i = 0; i < 3; i++) {
		valid &= validateStateValue(state.ownInfantry[i], "ownInfantry");
		valid &= validateStateValue(state.ownVehicles[i], "ownVehicles");
		valid &= validateStateValue(state.ownAircraft[i], "ownAircraft");
		valid &= validateStateValue(state.ownStructures[i], "ownStructures");
		valid &= validateStateValue(state.enemyInfantry[i], "enemyInfantry");
		valid &= validateStateValue(state.enemyVehicles[i], "enemyVehicles");
		valid &= validateStateValue(state.enemyAircraft[i], "enemyAircraft");
		valid &= validateStateValue(state.enemyStructures[i], "enemyStructures");
	}

	valid &= validateStateValue(state.gameTimeMinutes, "gameTimeMinutes");
	valid &= validateStateValue(state.techLevel, "techLevel");
	valid &= validateStateValue(state.baseThreat, "baseThreat");
	valid &= validateStateValue(state.armyStrength, "armyStrength");
	valid &= validateStateValue(state.underAttack, "underAttack");
	valid &= validateStateValue(state.distanceToEnemy, "distanceToEnemy");

	return valid;
}

Bool MLBridge::sendState(const MLGameState& state)
{
	if (!m_connected) {
		// Try to connect
		if (!connect()) {
			return false;
		}
	}

	// FIX R1: Validate state before sending
	if (!validateState(state)) {
		DEBUG_LOG(("MLBridge: State validation failed, skipping send\n"));
		return false;
	}

	AsciiString json = stateToJson(state);
	return writeMessage(json.str(), json.getLength());
}

Bool MLBridge::sendGameEnd(Bool victory, Real gameTimeMinutes, Real finalArmyStrength)
{
	if (!m_connected) {
		return false;
	}

	// Send game-end notification as JSON
	// FIX: Use snprintf with bounds checking to prevent buffer overflow
	char buffer[256];
	int written = snprintf(buffer, sizeof(buffer),
		"{"
		"\"type\":\"game_end\","
		"\"victory\":%s,"
		"\"game_time\":%.2f,"
		"\"army_strength\":%.2f"
		"}",
		victory ? "true" : "false",
		gameTimeMinutes,
		finalArmyStrength
	);

	if (written < 0 || written >= (int)sizeof(buffer)) {
		DEBUG_LOG(("MLBridge: sendGameEnd buffer overflow\n"));
		return false;
	}

	Bool result = writeMessage(buffer, strlen(buffer));
	DEBUG_LOG(("MLBridge: Sent game end notification (victory=%d)\n", victory));
	return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Recommendation Parsing
///////////////////////////////////////////////////////////////////////////////////////////////////

// Simple JSON number parser (finds "key":value and returns value)
// FIX: Added isfinite() validation to reject NaN/Inf values
static Real parseJsonFloat(const char* json, const char* key, Real defaultValue)
{
	char searchKey[64];
	snprintf(searchKey, sizeof(searchKey), "\"%s\":", key);

	const char* pos = strstr(json, searchKey);
	if (!pos) return defaultValue;

	pos += strlen(searchKey);
	while (*pos == ' ') pos++;

	Real value = (Real)atof(pos);

	// Validate the parsed value is finite (not NaN or Inf)
	if (!isfinite(value)) {
		DEBUG_LOG(("MLBridge: parseJsonFloat got non-finite value for key '%s', using default\n", key));
		return defaultValue;
	}

	return value;
}

static Int parseJsonInt(const char* json, const char* key, Int defaultValue)
{
	char searchKey[64];
	snprintf(searchKey, sizeof(searchKey), "\"%s\":", key);

	const char* pos = strstr(json, searchKey);
	if (!pos) return defaultValue;

	pos += strlen(searchKey);
	while (*pos == ' ') pos++;

	// FIX: Add validation matching parseJsonFloat pattern
	// Check for valid integer start (digit or minus sign)
	if (!(*pos == '-' || (*pos >= '0' && *pos <= '9'))) {
		DEBUG_LOG(("MLBridge: parseJsonInt invalid format for '%s'\n", key));
		return defaultValue;
	}

	// Use strtol for better error handling and range checking
	char* endptr = NULL;
	long value = strtol(pos, &endptr, 10);

	// Check for conversion failure
	if (endptr == pos) {
		DEBUG_LOG(("MLBridge: parseJsonInt conversion failed for '%s'\n", key));
		return defaultValue;
	}

	// Check for overflow (Int is typically 32-bit)
	if (value > 2147483647L || value < -2147483648L) {
		DEBUG_LOG(("MLBridge: parseJsonInt overflow for '%s'\n", key));
		return defaultValue;
	}

	return (Int)value;
}

// Clamp value to valid range [minVal, maxVal]
static Real clampReal(Real val, Real minVal, Real maxVal)
{
	if (val < minVal) return minVal;
	if (val > maxVal) return maxVal;
	return val;
}

// Parse JSON boolean value (finds "key":true/false and returns value)
static Bool parseJsonBool(const char* json, const char* key, Bool defaultValue)
{
	char searchKey[64];
	snprintf(searchKey, sizeof(searchKey), "\"%s\":", key);

	const char* pos = strstr(json, searchKey);
	if (!pos) return defaultValue;

	pos += strlen(searchKey);
	while (*pos == ' ') pos++;

	if (strncmp(pos, "true", 4) == 0) return true;
	if (strncmp(pos, "false", 5) == 0) return false;

	return defaultValue;
}

Bool MLBridge::parseRecommendation(const char* json, MLRecommendation& outRec)
{
	outRec.clear();

	// Parse and validate build priorities (must be in [0, 1])
	outRec.priorityEconomy = clampReal(parseJsonFloat(json, "priority_economy", 0.25f), 0.0f, 1.0f);
	outRec.priorityDefense = clampReal(parseJsonFloat(json, "priority_defense", 0.25f), 0.0f, 1.0f);
	outRec.priorityMilitary = clampReal(parseJsonFloat(json, "priority_military", 0.25f), 0.0f, 1.0f);
	outRec.priorityTech = clampReal(parseJsonFloat(json, "priority_tech", 0.25f), 0.0f, 1.0f);

	// Parse and validate army preferences (must be in [0, 1])
	outRec.preferInfantry = clampReal(parseJsonFloat(json, "prefer_infantry", 0.33f), 0.0f, 1.0f);
	outRec.preferVehicles = clampReal(parseJsonFloat(json, "prefer_vehicles", 0.34f), 0.0f, 1.0f);
	outRec.preferAircraft = clampReal(parseJsonFloat(json, "prefer_aircraft", 0.33f), 0.0f, 1.0f);

	// Parse and validate aggression (must be in [0, 1])
	outRec.aggression = clampReal(parseJsonFloat(json, "aggression", 0.5f), 0.0f, 1.0f);

	// Parse target (validate range)
	Int target = parseJsonInt(json, "target_player", -1);
	outRec.targetPlayer = (target >= -1 && target < 8) ? target : -1;

	outRec.valid = true;

	// Parse capabilities and update batched mode
	MLCapabilities caps;
	if (parseCapabilities(json, caps)) {
		m_serverCapabilities = caps;
		m_batchedModeEnabled = caps.hierarchical;
		DEBUG_LOG(("MLBridge: Capabilities updated - hierarchical=%d, tactical=%d, micro=%d\n",
			caps.hierarchical, caps.tactical, caps.micro));
	}

	return true;
}

Bool MLBridge::parseCapabilities(const char* json, MLCapabilities& outCaps)
{
	// Look for "capabilities" object
	const char* capsStart = strstr(json, "\"capabilities\"");
	if (!capsStart) return false;

	// Find the opening brace of the capabilities object
	const char* braceStart = strchr(capsStart, '{');
	if (!braceStart) return false;

	// Parse within the capabilities object (search from capsStart, not whole json)
	outCaps.hierarchical = parseJsonBool(capsStart, "hierarchical", false);
	outCaps.tactical = parseJsonBool(capsStart, "tactical", false);
	outCaps.micro = parseJsonBool(capsStart, "micro", false);

	return true;
}

Bool MLBridge::receiveRecommendation(MLRecommendation& outRec)
{
	if (!m_connected) {
		return false;
	}

	UnsignedInt length = 0;
	if (!readMessage(m_readBuffer, READ_BUFFER_SIZE - 1, length)) {
		return false;
	}

	if (parseRecommendation(m_readBuffer, outRec)) {
		m_lastRecommendation = outRec;
		m_hasRecommendation = true;
		m_lastRecommendationFrame = TheGameLogic ? TheGameLogic->getFrame() : 0;
		return true;
	}

	return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Recommendation Staleness Check
///////////////////////////////////////////////////////////////////////////////////////////////////

Bool MLBridge::isRecommendationStale() const
{
	if (!m_hasRecommendation) {
		return true;  // No recommendation yet = stale
	}

	UnsignedInt currentFrame = TheGameLogic ? TheGameLogic->getFrame() : 0;
	return (currentFrame - m_lastRecommendationFrame) > RECOMMENDATION_TIMEOUT_FRAMES;
}

// FIX I1: Check for extended staleness and trigger reconnection
Bool MLBridge::checkAndHandleStaleness()
{
	if (!m_hasRecommendation) {
		return false;  // Nothing to check yet
	}

	UnsignedInt currentFrame = TheGameLogic ? TheGameLogic->getFrame() : 0;
	UnsignedInt staleDuration = currentFrame - m_lastRecommendationFrame;

	// If stale for longer than alert threshold, log error and try to reconnect
	if (staleDuration > RECOMMENDATION_STALE_ALERT_FRAMES) {
		DEBUG_LOG(("MLBridge ERROR: Recommendation stale for %u frames (>%u), triggering reconnect\n",
			staleDuration, RECOMMENDATION_STALE_ALERT_FRAMES));

		// Disconnect to force reconnect on next attempt
		disconnect();
		return true;  // Indicates reconnection was triggered
	}

	return false;
}

MLRecommendation MLBridge::getValidRecommendation() const
{
	if (isRecommendationStale()) {
		// Return defaults when stale
		MLRecommendation defaults;
		defaults.clear();
		defaults.valid = false;  // Mark as invalid so AI uses parent logic
		return defaults;
	}
	return m_lastRecommendation;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Batched Protocol Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

void MLBatchedRequest::clear()
{
	frame = 0;
	playerIndex = 0;
	strategic.clear();
	numTeams = 0;
	numUnits = 0;
	memset(teamIds, 0, sizeof(teamIds));
	memset(unitIds, 0, sizeof(unitIds));
}

void MLBatchedResponse::clear()
{
	frame = 0;
	version = 0;
	strategic.clear();
	numTeamCommands = 0;
	numUnitCommands = 0;
}

void MLBridge::addTeamToBatch(MLBatchedRequest& request, Int teamId, const TacticalState& state)
{
	if (request.numTeams >= MAX_TEAMS_PER_BATCH) {
		DEBUG_LOG(("MLBridge: Team batch full, skipping team %d\n", teamId));
		return;
	}

	Int idx = request.numTeams++;
	request.teamIds[idx] = teamId;
	request.teamStates[idx] = state;
}

void MLBridge::addUnitToBatch(MLBatchedRequest& request, ObjectID unitId, const MicroState& state)
{
	if (request.numUnits >= MAX_UNITS_PER_BATCH) {
		DEBUG_LOG(("MLBridge: Unit batch full, skipping unit %d\n", unitId));
		return;
	}

	Int idx = request.numUnits++;
	request.unitIds[idx] = unitId;
	request.unitStates[idx] = state;
}

AsciiString MLBridge::batchedRequestToJson(const MLBatchedRequest& request)
{
	// Build JSON for batched request
	// Format:
	// {
	//   "frame": 1234,
	//   "player_id": 3,
	//   "strategic": { ... },
	//   "teams": [ {"id": 1, "state": [64 floats]}, ... ],
	//   "units": [ {"id": 101, "state": [32 floats]}, ... ]
	// }

	char* buffer = m_batchedReadBuffer;  // Reuse buffer for building
	size_t remaining = BATCHED_BUFFER_SIZE;
	char* pos = buffer;
	int written;

	// Header
	written = snprintf(pos, remaining,
		"{\"frame\":%u,\"player_id\":%d,",
		request.frame, request.playerIndex);
	if (written < 0 || (size_t)written >= remaining) return AsciiString("{}");
	pos += written; remaining -= written;

	// Strategic state (inline the important fields)
	written = snprintf(pos, remaining,
		"\"strategic\":{"
		"\"money\":%.2f,\"power\":%.2f,\"income\":%.2f,\"supply\":%.2f,"
		"\"own_infantry\":[%.2f,%.2f,%.2f],"
		"\"own_vehicles\":[%.2f,%.2f,%.2f],"
		"\"own_aircraft\":[%.2f,%.2f,%.2f],"
		"\"own_structures\":[%.2f,%.2f,%.2f],"
		"\"enemy_infantry\":[%.2f,%.2f,%.2f],"
		"\"enemy_vehicles\":[%.2f,%.2f,%.2f],"
		"\"enemy_aircraft\":[%.2f,%.2f,%.2f],"
		"\"enemy_structures\":[%.2f,%.2f,%.2f],"
		"\"game_time\":%.2f,\"tech_level\":%.2f,\"base_threat\":%.2f,"
		"\"army_strength\":%.2f,\"under_attack\":%.1f,\"distance_to_enemy\":%.2f,"
		"\"is_usa\":%.1f,\"is_china\":%.1f,\"is_gla\":%.1f}",
		request.strategic.money, request.strategic.powerBalance,
		request.strategic.incomeRate, request.strategic.supplyUsed,
		request.strategic.ownInfantry[0], request.strategic.ownInfantry[1], request.strategic.ownInfantry[2],
		request.strategic.ownVehicles[0], request.strategic.ownVehicles[1], request.strategic.ownVehicles[2],
		request.strategic.ownAircraft[0], request.strategic.ownAircraft[1], request.strategic.ownAircraft[2],
		request.strategic.ownStructures[0], request.strategic.ownStructures[1], request.strategic.ownStructures[2],
		request.strategic.enemyInfantry[0], request.strategic.enemyInfantry[1], request.strategic.enemyInfantry[2],
		request.strategic.enemyVehicles[0], request.strategic.enemyVehicles[1], request.strategic.enemyVehicles[2],
		request.strategic.enemyAircraft[0], request.strategic.enemyAircraft[1], request.strategic.enemyAircraft[2],
		request.strategic.enemyStructures[0], request.strategic.enemyStructures[1], request.strategic.enemyStructures[2],
		request.strategic.gameTimeMinutes, request.strategic.techLevel, request.strategic.baseThreat,
		request.strategic.armyStrength, request.strategic.underAttack, request.strategic.distanceToEnemy,
		request.strategic.isUSA, request.strategic.isChina, request.strategic.isGLA);
	if (written < 0 || (size_t)written >= remaining) return AsciiString("{}");
	pos += written; remaining -= written;

	// Teams array
	if (request.numTeams > 0) {
		written = snprintf(pos, remaining, ",\"teams\":[");
		if (written < 0 || (size_t)written >= remaining) return AsciiString("{}");
		pos += written; remaining -= written;

		for (Int i = 0; i < request.numTeams; i++) {
			if (i > 0) {
				written = snprintf(pos, remaining, ",");
				if (written < 0 || (size_t)written >= remaining) return AsciiString("{}");
				pos += written; remaining -= written;
			}

			// Serialize team state as array of floats
			Real stateArray[TacticalState::DIM];
			request.teamStates[i].toFloatArray(stateArray);

			written = snprintf(pos, remaining, "{\"id\":%d,\"state\":[", request.teamIds[i]);
			if (written < 0 || (size_t)written >= remaining) return AsciiString("{}");
			pos += written; remaining -= written;

			for (UnsignedInt j = 0; j < TacticalState::DIM; j++) {
				written = snprintf(pos, remaining, j > 0 ? ",%.3f" : "%.3f", stateArray[j]);
				if (written < 0 || (size_t)written >= remaining) return AsciiString("{}");
				pos += written; remaining -= written;
			}

			written = snprintf(pos, remaining, "]}");
			if (written < 0 || (size_t)written >= remaining) return AsciiString("{}");
			pos += written; remaining -= written;
		}

		written = snprintf(pos, remaining, "]");
		if (written < 0 || (size_t)written >= remaining) return AsciiString("{}");
		pos += written; remaining -= written;
	}

	// Units array
	if (request.numUnits > 0) {
		written = snprintf(pos, remaining, ",\"units\":[");
		if (written < 0 || (size_t)written >= remaining) return AsciiString("{}");
		pos += written; remaining -= written;

		for (Int i = 0; i < request.numUnits; i++) {
			if (i > 0) {
				written = snprintf(pos, remaining, ",");
				if (written < 0 || (size_t)written >= remaining) return AsciiString("{}");
				pos += written; remaining -= written;
			}

			// Serialize unit state as array of floats
			Real stateArray[MicroState::DIM];
			request.unitStates[i].toFloatArray(stateArray);

			written = snprintf(pos, remaining, "{\"id\":%u,\"state\":[", request.unitIds[i]);
			if (written < 0 || (size_t)written >= remaining) return AsciiString("{}");
			pos += written; remaining -= written;

			for (UnsignedInt j = 0; j < MicroState::DIM; j++) {
				written = snprintf(pos, remaining, j > 0 ? ",%.3f" : "%.3f", stateArray[j]);
				if (written < 0 || (size_t)written >= remaining) return AsciiString("{}");
				pos += written; remaining -= written;
			}

			written = snprintf(pos, remaining, "]}");
			if (written < 0 || (size_t)written >= remaining) return AsciiString("{}");
			pos += written; remaining -= written;
		}

		written = snprintf(pos, remaining, "]");
		if (written < 0 || (size_t)written >= remaining) return AsciiString("{}");
		pos += written; remaining -= written;
	}

	// Close object
	written = snprintf(pos, remaining, "}");
	if (written < 0 || (size_t)written >= remaining) return AsciiString("{}");

	return AsciiString(buffer);
}

Bool MLBridge::sendBatchedState(const MLBatchedRequest& request)
{
	if (!m_connected) {
		if (!connect()) {
			return false;
		}
	}

	AsciiString json = batchedRequestToJson(request);
	return writeMessage(json.str(), json.getLength());
}

Bool MLBridge::parseTacticalCommand(const char* json, TacticalCommand& outCmd)
{
	outCmd.clear();

	outCmd.teamId = parseJsonInt(json, "id", -1);
	if (outCmd.teamId < 0) return false;

	Int actionInt = parseJsonInt(json, "action", TACTICAL_HOLD);
	if (actionInt < 0 || actionInt >= TACTICAL_ACTION_COUNT) {
		actionInt = TACTICAL_HOLD;
	}
	outCmd.action = (TacticalActionType)actionInt;

	outCmd.targetX = clampReal(parseJsonFloat(json, "x", 0.5f), 0.0f, 1.0f);
	outCmd.targetY = clampReal(parseJsonFloat(json, "y", 0.5f), 0.0f, 1.0f);
	outCmd.attitude = clampReal(parseJsonFloat(json, "attitude", 0.5f), 0.0f, 1.0f);

	outCmd.valid = true;
	return true;
}

Bool MLBridge::parseMicroCommand(const char* json, MicroCommand& outCmd)
{
	outCmd.clear();

	outCmd.unitId = (ObjectID)parseJsonInt(json, "id", INVALID_ID);
	if (outCmd.unitId == INVALID_ID) return false;

	Int actionInt = parseJsonInt(json, "action", MICRO_FOLLOW_TEAM);
	if (actionInt < 0 || actionInt >= MICRO_ACTION_COUNT) {
		actionInt = MICRO_FOLLOW_TEAM;
	}
	outCmd.action = (MicroActionType)actionInt;

	// Angle is in radians, -pi to pi, normalized to -1 to 1 in transmission
	outCmd.moveAngle = clampReal(parseJsonFloat(json, "angle", 0.0f), -1.0f, 1.0f) * 3.14159265f;
	outCmd.moveDistance = clampReal(parseJsonFloat(json, "dist", 0.0f), 0.0f, 1.0f);

	outCmd.valid = true;
	return true;
}

// Find next JSON object in array starting from position
static const char* findNextArrayObject(const char* start)
{
	const char* pos = start;
	while (*pos && *pos != '{') pos++;
	if (*pos != '{') return NULL;
	return pos;
}

// Find end of JSON object (matching closing brace)
static const char* findObjectEnd(const char* start)
{
	if (*start != '{') return NULL;

	const char* pos = start + 1;
	int depth = 1;

	while (*pos && depth > 0) {
		if (*pos == '{') depth++;
		else if (*pos == '}') depth--;
		pos++;
	}

	return depth == 0 ? pos : NULL;
}

Bool MLBridge::parseBatchedResponse(const char* json, MLBatchedResponse& outResponse)
{
	outResponse.clear();

	// Parse frame and version
	outResponse.frame = (UnsignedInt)parseJsonInt(json, "frame", 0);
	outResponse.version = parseJsonInt(json, "version", 0);

	// Parse capabilities and update batched mode (do this before strategic to set mode early)
	MLCapabilities caps;
	if (parseCapabilities(json, caps)) {
		m_serverCapabilities = caps;
		m_batchedModeEnabled = caps.hierarchical;
	}

	// Parse strategic recommendation
	const char* strategicStart = strstr(json, "\"strategic\":");
	if (strategicStart) {
		strategicStart = strchr(strategicStart, '{');
		if (strategicStart) {
			// Note: parseRecommendation also tries to parse capabilities, but that's harmless
			parseRecommendation(strategicStart, outResponse.strategic);
		}
	}

	// Parse team commands
	const char* teamsStart = strstr(json, "\"teams\":");
	if (teamsStart) {
		teamsStart = strchr(teamsStart, '[');
		if (teamsStart) {
			const char* pos = teamsStart + 1;
			while ((pos = findNextArrayObject(pos)) != NULL) {
				if (outResponse.numTeamCommands >= MAX_TEAMS_PER_BATCH) break;

				const char* objEnd = findObjectEnd(pos);
				if (!objEnd) break;

				// Extract object as substring for parsing
				size_t objLen = objEnd - pos;
				if (objLen < BATCHED_BUFFER_SIZE - 1) {
					char objBuffer[1024];
					if (objLen < sizeof(objBuffer) - 1) {
						strncpy(objBuffer, pos, objLen);
						objBuffer[objLen] = '\0';

						TacticalCommand cmd;
						if (parseTacticalCommand(objBuffer, cmd)) {
							outResponse.teamCommands[outResponse.numTeamCommands++] = cmd;
						}
					}
				}

				pos = objEnd;
			}
		}
	}

	// Parse unit commands
	const char* unitsStart = strstr(json, "\"units\":");
	if (unitsStart) {
		unitsStart = strchr(unitsStart, '[');
		if (unitsStart) {
			const char* pos = unitsStart + 1;
			while ((pos = findNextArrayObject(pos)) != NULL) {
				if (outResponse.numUnitCommands >= MAX_UNITS_PER_BATCH) break;

				const char* objEnd = findObjectEnd(pos);
				if (!objEnd) break;

				// Extract object as substring for parsing
				size_t objLen = objEnd - pos;
				if (objLen < 1024) {
					char objBuffer[1024];
					strncpy(objBuffer, pos, objLen);
					objBuffer[objLen] = '\0';

					MicroCommand cmd;
					if (parseMicroCommand(objBuffer, cmd)) {
						outResponse.unitCommands[outResponse.numUnitCommands++] = cmd;
					}
				}

				pos = objEnd;
			}
		}
	}

	return true;
}

Bool MLBridge::receiveBatchedResponse(MLBatchedResponse& outResponse)
{
	if (!m_connected) {
		return false;
	}

	UnsignedInt length = 0;
	if (!readMessage(m_batchedReadBuffer, BATCHED_BUFFER_SIZE - 1, length)) {
		return false;
	}

	if (parseBatchedResponse(m_batchedReadBuffer, outResponse)) {
		m_lastBatchedResponse = outResponse;
		m_hasBatchedResponse = true;

		// Also update strategic recommendation for backward compatibility
		m_lastRecommendation = outResponse.strategic;
		m_hasRecommendation = true;
		m_lastRecommendationFrame = TheGameLogic ? TheGameLogic->getFrame() : 0;

		return true;
	}

	return false;
}
