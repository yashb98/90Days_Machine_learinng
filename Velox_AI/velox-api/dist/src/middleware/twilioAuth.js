"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.validateTwilioWebhook = void 0;
const twilio_1 = require("twilio");
const logger_1 = require("../utils/logger");
const validateTwilioWebhook = (req, res, next) => {
    // Skip validation in local dev if we are just testing with Postman/Curl
    if (process.env.NODE_ENV !== "production" && !req.headers["x-twilio-signature"]) {
        logger_1.logger.warn("Skipping Twilio validation (No Signature Header)");
        return next();
    }
    const authToken = process.env.TWILIO_AUTH_TOKEN;
    const signature = req.headers["x-twilio-signature"];
    // The URL Twilio thinks it is hitting (important when using Ngrok)
    // We'll set a custom header or use the Host header
    const proto = req.headers["x-forwarded-proto"] || "http";
    const url = `${proto}://${req.headers.host}${req.originalUrl}`;
    const params = req.body;
    if (!authToken) {
        logger_1.logger.error("TWILIO_AUTH_TOKEN is missing in .env");
        return res.status(500).send("Server Config Error");
    }
    const isValid = (0, twilio_1.validateRequest)(authToken, signature, url, params);
    if (isValid) {
        next();
    }
    else {
        logger_1.logger.warn(` Rejected invalid Twilio request from ${req.ip}`);
        res.status(403).send("Forbidden");
    }
};
exports.validateTwilioWebhook = validateTwilioWebhook;
