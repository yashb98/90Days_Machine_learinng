import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
// import stdio = require("@modelcontextprotocol/sdk/server/stdio.js")
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
    name: 'test',
    version: "1.0.0",
    capabilities: {
        resources:{},
        tools: {},
        prompts: {},
    }
})

async function main(){
    const transport = new StdioServerTransport()
    await server.connect(transport)
} 

main()