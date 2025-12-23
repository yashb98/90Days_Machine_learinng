// List available resources
export async function listResources(client, metadata) {
    try {
        const params = metadata && Object.keys(metadata).length > 0 ? { _meta: metadata } : {};
        const response = await client.listResources(params);
        return response;
    }
    catch (error) {
        throw new Error(`Failed to list resources: ${error instanceof Error ? error.message : String(error)}`);
    }
}
// Read a resource
export async function readResource(client, uri, metadata) {
    try {
        const params = { uri };
        if (metadata && Object.keys(metadata).length > 0) {
            params._meta = metadata;
        }
        const response = await client.readResource(params);
        return response;
    }
    catch (error) {
        throw new Error(`Failed to read resource ${uri}: ${error instanceof Error ? error.message : String(error)}`);
    }
}
// List resource templates
export async function listResourceTemplates(client, metadata) {
    try {
        const params = metadata && Object.keys(metadata).length > 0 ? { _meta: metadata } : {};
        const response = await client.listResourceTemplates(params);
        return response;
    }
    catch (error) {
        throw new Error(`Failed to list resource templates: ${error instanceof Error ? error.message : String(error)}`);
    }
}
