import { Request, Response, NextFunction } from "express";
import { validateRequest } from "twilio";
import { logger } from "../utils/logger";

export const validateTwilioWebhook = (req: Request, res: Response, next: NextFunction) => {
  // Skip validation in local dev if we are just testing with Postman/Curl
  if (process.env.NODE_ENV !== "production" && !req.headers["x-twilio-signature"]) {
    logger.warn("Skipping Twilio validation (No Signature Header)");
    return next();
  }

  const authToken = process.env.TWILIO_AUTH_TOKEN;
  const signature = req.headers["x-twilio-signature"] as string;
  
  // The URL Twilio thinks it is hitting (important when using Ngrok)
  // We'll set a custom header or use the Host header
  const proto = req.headers["x-forwarded-proto"] || "http";
  const url = `${proto}://${req.headers.host}${req.originalUrl}`;
  const params = req.body;

  if (!authToken) {
    logger.error("TWILIO_AUTH_TOKEN is missing in .env");
    return res.status(500).send("Server Config Error");
  }

  const isValid = validateRequest(authToken, signature, url, params);

  if (isValid) {
    next();
  } else {
    logger.warn(` Rejected invalid Twilio request from ${req.ip}`);
    res.status(403).send("Forbidden");
  }
};