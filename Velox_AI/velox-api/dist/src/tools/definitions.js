"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.tools = void 0;
// Using 'any' type to avoid ESM/CommonJS import conflicts
// The actual tool implementations are in the registry
exports.tools = [
    {
        name: "check_order_status",
        description: "Look up the current status of a customer's order using their Order ID.",
        parametersJsonSchema: {
            type: "object",
            properties: {
                order_id: {
                    type: "string",
                    description: "The order ID (e.g., ORD-12345)",
                },
            },
            required: ["order_id"],
        },
    },
    {
        name: "check_item_stock",
        description: "Check if an item is available in the warehouse.",
        parametersJsonSchema: {
            type: "object",
            properties: {
                item_name: {
                    type: "string",
                    description: "The name of the product",
                },
            },
            required: ["item_name"],
        },
    },
];
