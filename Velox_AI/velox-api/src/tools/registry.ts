import { logger } from "../utils/logger";

// Mock Database
const MOCK_ORDERS: Record<string, string> = {
  "123": "Shipped - Arriving Tuesday",
  "456": "Processing",
  "999": "Cancelled",
};

const MOCK_STOCK: Record<string, number> = {
  "laptop": 5,
  "mouse": 0,
  "keyboard": 12
};

export const toolRegistry = {
  check_order_status: async (args: { order_id: string }) => {
    logger.info(`🛠️ Tool Triggered: check_order_status for ${args.order_id}`);
    const status = MOCK_ORDERS[args.order_id] || "Order not found";
    return { status };
  },

  check_item_stock: async (args: { item_name: string }) => {
    logger.info(`🛠️ Tool Triggered: check_item_stock for ${args.item_name}`);
    const stock = MOCK_STOCK[args.item_name.toLowerCase()] ?? "Unknown item";
    return { stock };
  }
};