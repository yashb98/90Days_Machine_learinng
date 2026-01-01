// velox-api/src/server.ts
import { app, logger } from "./app.js";

const PORT = process.env.PORT || 8080; // Cloud Run defaults to 8080

const server = app.listen(PORT, () => {
  logger.info(`🚀 Server listening on port ${PORT}`);
});

// Graceful Shutdown (Handle Cloud Run SIGTERM)
process.on("SIGTERM", () => {
  logger.info("SIGTERM received. Shutting down gracefully...");
  server.close(() => {
    logger.info("Server closed.");
    process.exit(0);
  });
});