"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
// velox-api/src/server.ts
const app_js_1 = require("./app.js");
const PORT = process.env.PORT || 8080; // Cloud Run defaults to 8080
const server = app_js_1.app.listen(PORT, () => {
    app_js_1.logger.info(`🚀 Server listening on port ${PORT}`);
});
// Graceful Shutdown (Handle Cloud Run SIGTERM)
process.on("SIGTERM", () => {
    app_js_1.logger.info("SIGTERM received. Shutting down gracefully...");
    server.close(() => {
        app_js_1.logger.info("Server closed.");
        process.exit(0);
    });
});
