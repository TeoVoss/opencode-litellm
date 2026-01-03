import { Global } from "../global"
import { Log } from "../util/log"
import path from "path"
import z from "zod"
import { Auth } from "../auth"
import { Config } from "../config/config"
import { Env } from "../env"

export namespace LiteLLM {
  const log = Log.create({ service: "litellm" })
  const filepath = path.join(Global.Path.cache, "litellm-models.json")

  // OpenAI-compatible model response
  export const ModelResponse = z.object({
    id: z.string(),
    object: z.string(),
    created: z.number(),
    owned_by: z.string(),
  })

  export const ModelsResponse = z.object({
    data: z.array(ModelResponse),
    object: z.string(),
  })

  export type ModelResponse = z.infer<typeof ModelResponse>

  // Model capability detection based on model ID
  function detectCapabilities(modelId: string) {
    const id = modelId.toLowerCase()

    // Reasoning models
    const reasoning =
      id.includes("o1") ||
      id.includes("o3") ||
      id.includes("o4") ||
      id.includes("reasoner") ||
      id.includes("r1") ||
      id.includes("thinking") ||
      id.includes("opus") ||
      id.includes("gpt-5")

    // Image input capability
    const imageInput =
      id.includes("vision") ||
      id.includes("4o") ||
      id.includes("gpt-5") ||
      id.includes("claude") ||
      id.includes("gemini") ||
      id.includes("image")

    // Tool call capability - most chat models support it
    const toolCall =
      !id.includes("embedding") &&
      !id.includes("whisper") &&
      !id.includes("tts") &&
      !id.includes("dall-e") &&
      !id.includes("image-preview")

    // PDF support (Claude and some others)
    const pdfInput = id.includes("claude") || id.includes("gemini")

    return {
      temperature: !reasoning,
      reasoning,
      attachment: imageInput || pdfInput,
      toolcall: toolCall,
      input: {
        text: true,
        audio: id.includes("whisper") || id.includes("audio"),
        image: imageInput,
        video: id.includes("gemini"),
        pdf: pdfInput,
      },
      output: {
        text: true,
        audio: id.includes("tts") || id.includes("audio"),
        image: id.includes("dall-e") || id.includes("image"),
        video: false,
        pdf: false,
      },
      interleaved: reasoning,
    }
  }

  // Estimate context limits based on model ID
  function estimateLimits(modelId: string) {
    const id = modelId.toLowerCase()

    if (id.includes("claude")) {
      return { context: 200000, output: 64000 }
    }
    if (id.includes("gemini-3")) {
      return { context: 1000000, output: 65536 }
    }
    if (id.includes("gemini-2.5")) {
      return { context: 1048576, output: 65536 }
    }
    if (id.includes("gpt-5")) {
      return { context: 1047552, output: 100000 }
    }
    if (id.includes("gpt-4o")) {
      return { context: 128000, output: 16384 }
    }
    if (id.includes("deepseek")) {
      return { context: 128000, output: 8192 }
    }
    // Default
    return { context: 128000, output: 8192 }
  }

  // Get human-readable model name with official formatting
  function getModelName(modelId: string): string {
    // Remove provider prefix (e.g., "deepseek/deepseek-v3" -> "deepseek-v3")
    const parts = modelId.split("/")
    let name = parts[parts.length - 1]

    // Handle "global." prefix first (e.g., "global.anthropic.claude-...")
    if (name.startsWith("global.")) {
      name = name.slice(7) // Remove "global."
    }

    // Handle Bedrock-style Claude model IDs (e.g., "anthropic.claude-haiku-4-5-20251001-v1:0")
    if (name.includes(".") && (name.includes("claude") || name.includes("anthropic"))) {
      const dotParts = name.split(".")
      if (dotParts.length > 1) {
        name = dotParts.slice(1).join(".")
      }
      // Remove version suffix like "-v1:0" and date suffix like "-20251001"
      name = name.replace(/-v\d+:\d+$/, "").replace(/-\d{8}$/, "")

      // Parse Claude model: "claude-haiku-4-5" -> "Claude 4.5 Haiku"
      const claudeMatch = name.match(/^claude-(\w+)-(\d+)-(\d+)$/)
      if (claudeMatch) {
        const [, variant, major, minor] = claudeMatch
        const variantName = variant.charAt(0).toUpperCase() + variant.slice(1)
        return `Claude ${major}.${minor} ${variantName}`
      }
    }

    // Handle DeepSeek models
    if (name.startsWith("deepseek-") || name.startsWith("deepseek/")) {
      name = name.replace(/^deepseek[-/]/, "")
      const deepseekNames: Record<string, string> = {
        chat: "DeepSeek Chat",
        coder: "DeepSeek Coder",
        reasoner: "DeepSeek Reasoner",
        r1: "DeepSeek R1",
        v3: "DeepSeek V3",
        "v3.2": "DeepSeek V3.2",
      }
      if (deepseekNames[name]) return deepseekNames[name]
      return `DeepSeek ${name.charAt(0).toUpperCase() + name.slice(1)}`
    }

    // Handle GPT models - uppercase GPT
    if (name.startsWith("gpt-")) {
      // "gpt-4o" -> "GPT-4o", "gpt-5.2" -> "GPT-5.2", "gpt-5.1-codex" -> "GPT-5.1 Codex"
      const gptPart = name.slice(4) // Remove "gpt-"
      const gptParts = gptPart.split("-")
      const version = gptParts[0] // "4o", "5.2", "5.1"
      const suffix = gptParts.slice(1).map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ")
      return suffix ? `GPT-${version} ${suffix}` : `GPT-${version}`
    }

    // Handle Gemini models
    if (name.startsWith("gemini-")) {
      // "gemini-2.5-flash" -> "Gemini 2.5 Flash"
      return name
        .split("-")
        .map((word, i) => {
          if (i === 0) return "Gemini"
          if (/^\d/.test(word)) return word // Keep version numbers as-is
          return word.charAt(0).toUpperCase() + word.slice(1)
        })
        .join(" ")
    }

    // Default: title case with space separator
    return name
      .split("-")
      .map((word) => {
        if (/^\d/.test(word) || /^v\d/.test(word)) return word
        return word.charAt(0).toUpperCase() + word.slice(1)
      })
      .join(" ")
  }

  // Determine model family - use unique family per model so all models show in UI
  // (UI only shows latest model per family by default)
  function getModelFamily(modelId: string): string {
    // Use the model ID itself as family to ensure all models are visible
    // This bypasses the "latest per family" filter in the UI
    return modelId
  }

  // Filter out non-chat models
  function isChatModel(modelId: string): boolean {
    const id = modelId.toLowerCase()
    // Exclude embedding, audio-only, image-generation models
    return (
      !id.includes("embedding") &&
      !id.includes("whisper") &&
      !id.includes("tts") &&
      !id.includes("dall-e") &&
      !id.includes("ada-002") &&
      !id.includes("*") // exclude wildcard models
    )
  }

  export interface LiteLLMConfig {
    baseURL: string
    apiKey: string
  }

  async function getConfig(): Promise<LiteLLMConfig | undefined> {
    // Check environment variables first (works without context)
    const envKey = process.env.LITELLM_API_KEY
    const envUrl = process.env.LITELLM_BASE_URL ?? process.env.LITELLM_API_BASE
    if (envKey) {
      return {
        baseURL: envUrl ?? "https://ai-gateway.wepieoa.com",
        apiKey: envKey,
      }
    }

    // Try to get config from context (may fail in standalone mode)
    try {
      // Check auth store
      const auth = await Auth.get("litellm")
      if (auth?.type === "api") {
        const config = await Config.get()
        const providerConfig = config.provider?.["litellm"]
        const baseURL = providerConfig?.options?.baseURL ?? providerConfig?.api ?? "https://ai-gateway.wepieoa.com"
        return {
          baseURL,
          apiKey: auth.key,
        }
      }

      // Check Env helper (requires context)
      const contextEnvKey = Env.get("LITELLM_API_KEY")
      const contextEnvUrl = Env.get("LITELLM_BASE_URL") ?? Env.get("LITELLM_API_BASE")
      if (contextEnvKey) {
        return {
          baseURL: contextEnvUrl ?? "https://ai-gateway.wepieoa.com",
          apiKey: contextEnvKey,
        }
      }

      // Check config
      const config = await Config.get()
      const providerConfig = config.provider?.["litellm"]
      if (providerConfig?.options?.apiKey) {
        return {
          baseURL: providerConfig.options.baseURL ?? providerConfig.api ?? "https://ai-gateway.wepieoa.com",
          apiKey: providerConfig.options.apiKey,
        }
      }
    } catch (e) {
      // Context not available, fall back to env vars only
      log.debug("context not available, using env vars only", { error: e })
    }

    return undefined
  }

  // Normalize baseURL to not include /v1 suffix
  function normalizeBaseURL(baseURL: string): string {
    let url = baseURL.replace(/\/$/, "") // Remove trailing slash
    // Remove /v1 suffix if present (we'll add it when needed)
    if (url.endsWith("/v1")) {
      url = url.slice(0, -3)
    }
    return url
  }

  export async function fetchModels(config: LiteLLMConfig): Promise<ModelResponse[]> {
    const baseURL = normalizeBaseURL(config.baseURL)
    const url = `${baseURL}/v1/models`
    log.info("fetching models", { url })

    const response = await fetch(url, {
      headers: {
        Authorization: `Bearer ${config.apiKey}`,
        "Content-Type": "application/json",
      },
      signal: AbortSignal.timeout(30 * 1000),
    })

    if (!response.ok) {
      throw new Error(`Failed to fetch LiteLLM models: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    const parsed = ModelsResponse.safeParse(data)

    if (!parsed.success) {
      log.error("Invalid models response", { error: parsed.error })
      throw new Error("Invalid models response from LiteLLM")
    }

    return parsed.data.data
  }

  export interface LiteLLMProvider {
    id: string
    name: string
    api: string
    env: string[]
    npm: string
    models: Record<
      string,
      {
        id: string
        name: string
        family?: string
        release_date: string
        attachment: boolean
        reasoning: boolean
        temperature: boolean
        tool_call: boolean
        interleaved?: boolean | { field: "reasoning_content" | "reasoning_details" }
        cost?: {
          input: number
          output: number
          cache_read?: number
          cache_write?: number
        }
        limit: {
          context: number
          output: number
        }
        modalities?: {
          input: ("text" | "audio" | "image" | "video" | "pdf")[]
          output: ("text" | "audio" | "image" | "video" | "pdf")[]
        }
        options: Record<string, any>
        headers?: Record<string, string>
      }
    >
  }

  export async function get(): Promise<LiteLLMProvider | undefined> {
    const config = await getConfig()
    if (!config) {
      log.info("no litellm config found")
      return undefined
    }

    // Try to load from cache first
    const file = Bun.file(filepath)
    const cached = await file.json().catch(() => undefined)

    // Check if cache is fresh (less than 1 hour old)
    if (cached && cached.timestamp && Date.now() - cached.timestamp < 60 * 60 * 1000) {
      log.info("using cached litellm models")
      return cached.provider as LiteLLMProvider
    }

    // Fetch fresh models
    try {
      const models = await fetchModels(config)
      const chatModels = models.filter((m) => isChatModel(m.id))

      const normalizedBaseURL = normalizeBaseURL(config.baseURL)
      const provider: LiteLLMProvider = {
        id: "litellm",
        name: "LiteLLM",
        api: normalizedBaseURL + "/v1",
        env: ["LITELLM_API_KEY"],
        npm: "@ai-sdk/openai-compatible",
        models: {},
      }

      for (const model of chatModels) {
        const capabilities = detectCapabilities(model.id)
        const limits = estimateLimits(model.id)

        provider.models[model.id] = {
          id: model.id,
          name: getModelName(model.id),
          family: getModelFamily(model.id),
          release_date: new Date().toISOString().split("T")[0],
          attachment: capabilities.attachment,
          reasoning: capabilities.reasoning,
          temperature: capabilities.temperature,
          tool_call: capabilities.toolcall,
          interleaved: capabilities.interleaved,
          limit: limits,
          modalities: {
            input: Object.entries(capabilities.input)
              .filter(([, v]) => v)
              .map(([k]) => k as any),
            output: Object.entries(capabilities.output)
              .filter(([, v]) => v)
              .map(([k]) => k as any),
          },
          options: {},
        }
      }

      // Cache the result
      await Bun.write(
        file,
        JSON.stringify({
          timestamp: Date.now(),
          provider,
        }),
      )

      log.info("fetched litellm models", { count: Object.keys(provider.models).length })
      return provider
    } catch (e) {
      log.error("failed to fetch litellm models", { error: e })

      // Return cached data if available, even if stale
      if (cached?.provider) {
        log.info("using stale cached litellm models")
        return cached.provider as LiteLLMProvider
      }

      return undefined
    }
  }

  export async function refresh() {
    const config = await getConfig()
    if (!config) return

    try {
      await get()
    } catch (e) {
      log.error("failed to refresh litellm models", { error: e })
    }
  }
}

// Refresh models every hour
setInterval(() => LiteLLM.refresh(), 60 * 1000 * 60).unref()
