// List available prompts
export async function listPrompts(client, metadata) {
    try {
        const params = metadata && Object.keys(metadata).length > 0 ? { _meta: metadata } : {};
        const response = await client.listPrompts(params);
        return response;
    }
    catch (error) {
        throw new Error(`Failed to list prompts: ${error instanceof Error ? error.message : String(error)}`);
    }
}
// Get a prompt
export async function getPrompt(client, name, args, metadata) {
    try {
        // Convert all arguments to strings for prompt arguments
        const stringArgs = {};
        if (args) {
            for (const [key, value] of Object.entries(args)) {
                if (typeof value === "string") {
                    stringArgs[key] = value;
                }
                else if (value === null || value === undefined) {
                    stringArgs[key] = String(value);
                }
                else {
                    stringArgs[key] = JSON.stringify(value);
                }
            }
        }
        const params = {
            name,
            arguments: stringArgs,
        };
        if (metadata && Object.keys(metadata).length > 0) {
            params._meta = metadata;
        }
        const response = await client.getPrompt(params);
        return response;
    }
    catch (error) {
        throw new Error(`Failed to get prompt: ${error instanceof Error ? error.message : String(error)}`);
    }
}
