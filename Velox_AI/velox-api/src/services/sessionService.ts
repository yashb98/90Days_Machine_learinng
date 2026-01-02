import redis from "../config/redis";

// Enums ensure we don't make typos in critical logic
export enum CallStage {
  LISTENING = "listening",
  THINKING = "thinking",
  SPEAKING = "speaking",
  TOOL_EXECUTION = "tool_execution"
}

export class SessionService {
  private static EXPIRATION_SECONDS = 60 * 60; // Clean up 1 hour after call ends

  /**
   * Initializes a new call session with default values
   */
  static async initSession(callSid: string, agentId: string) {
    const key = `call:${callSid}`;
    await redis.hset(key, {
      stage: CallStage.LISTENING,
      sequence_id: 0,
      interrupt_count: 0,
      agent_id: agentId,
      start_time: Date.now()
    });
    // Auto-expire keys so Redis doesn't fill up with old calls
    await redis.expire(key, this.EXPIRATION_SECONDS);
  }

  /**
   * Atomic State Transition
   */
  static async setStage(callSid: string, stage: CallStage) {
    await redis.hset(`call:${callSid}`, "stage", stage);
  }

  /**
   * Get full state (for the AI Worker)
   */
  static async getSession(callSid: string) {
    return await redis.hgetall(`call:${callSid}`);
  }

  /**
   * Increment Sequence ID (for UDP audio ordering)
   * Returns the NEW value
   */
  static async incrementSequence(callSid: string) {
    return await redis.hincrby(`call:${callSid}`, "sequence_id", 1);
  }
}