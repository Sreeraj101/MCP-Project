from mcp.server.fastmcp import FastMCP
import httpx

# 1. Initialize the FastMCP server instance
mcp = FastMCP("DevOpsAssistant")

# 2. Expose a tool using the decorator
@mcp.tool()
async def fetch_github_repo_summary(owner: str, repo: str) -> str:
    """Fetch structured metadata for a public GitHub repository.

    Args:
        owner: The GitHub organization or username (e.g., 'psf')
        repo: The repository name (e.g., 'requests')
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"User-Agent": "MCP-DevOps-Assistant"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url, headers=headers)

            if response.status_code == 404:
                return f"Error: Repository '{owner}/{repo}' not found."

            response.raise_for_status()
            data = response.json()

            # Truncate and select key fields to prevent context window bloat
            return (
                f"Full Name: {data.get('full_name')}\n"
                f"Description: {data.get('description', 'No description provided')}\n"
                f"Stars: {data.get('stargazers_count')}\n"
                f"Forks: {data.get('forks_count')}\n"
                f"Open Issues: {data.get('open_issues_count')}\n"
                f"Primary Language: {data.get('language', 'Unknown')}"
            )
        except httpx.TimeoutException:
            return "Error: Request to GitHub API timed out."
        except httpx.RequestError as exc:
            return f"Network error occurred: {str(exc)}"

# 3. Standard entrypoint to run the server over stdio
if __name__ == "__main__":
    mcp.run(transport="stdio")