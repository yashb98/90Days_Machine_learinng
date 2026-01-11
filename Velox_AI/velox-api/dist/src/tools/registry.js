"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.toolRegistry = void 0;
const logger_1 = require("../utils/logger");
// Mock Database
const MOCK_ORDERS = {
    "123": "Shipped - Arriving Tuesday",
    "456": "Processing",
    "999": "Cancelled",
};
const MOCK_STOCK = {
    "laptop": 5,
    "mouse": 0,
    "keyboard": 12
};
exports.toolRegistry = {
    check_order_status: async (args) => {
        logger_1.logger.info(`🛠️ Tool Triggered: check_order_status for ${args.order_id}`);
        const status = MOCK_ORDERS[args.order_id] || "Order not found";
        return { status };
    },
    check_item_stock: async (args) => {
        logger_1.logger.info(`🛠️ Tool Triggered: check_item_stock for ${args.item_name}`);
        const stock = MOCK_STOCK[args.item_name.toLowerCase()] ?? "Unknown item";
        return { stock };
    }
};
