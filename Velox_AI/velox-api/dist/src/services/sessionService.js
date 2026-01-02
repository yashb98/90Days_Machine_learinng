"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.SessionService = exports.CallStage = void 0;
const redis_1 = __importDefault(require("../config/redis"));
// Enums ensure we don't make typos in critical logic
var CallStage;
(function (CallStage) {
    CallStage["LISTENING"] = "listening";
    CallStage["THINKING"] = "thinking";
    CallStage["SPEAKING"] = "speaking";
    CallStage["TOOL_EXECUTION"] = "tool_execution";
})(CallStage || (exports.CallStage = CallStage = {}));
class SessionService {
    /**
     * Initializes a new call session with default values
     */
    static async initSession(callSid, agentId) {
        const key = `call:${callSid}`;
        await redis_1.default.hset(key, {
            stage: CallStage.LISTENING,
            sequence_id: 0,
            interrupt_count: 0,
            agent_id: agentId,
            start_time: Date.now()
        });
        // Auto-expire keys so Redis doesn't fill up with old calls
        await redis_1.default.expire(key, this.EXPIRATION_SECONDS);
    }
    /**
     * Atomic State Transition
     */
    static async setStage(callSid, stage) {
        await redis_1.default.hset(`call:${callSid}`, "stage", stage);
    }
    /**
     * Get full state (for the AI Worker)
     */
    static async getSession(callSid) {
        return await redis_1.default.hgetall(`call:${callSid}`);
    }
    /**
     * Increment Sequence ID (for UDP audio ordering)
     * Returns the NEW value
     */
    static async incrementSequence(callSid) {
        return await redis_1.default.hincrby(`call:${callSid}`, "sequence_id", 1);
    }
}
exports.SessionService = SessionService;
SessionService.EXPIRATION_SECONDS = 60 * 60; // Clean up 1 hour after call ends
