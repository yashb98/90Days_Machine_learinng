
<div align="center">

**Enterprise-Grade AI Voice Agent Platform**

*Build, Deploy, and Manage Intelligent AI Agents for Voice and Chat Applications*

[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue.svg)](https://www.typescriptlang.org/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-7-646CFF.svg)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4-38B2AC.svg)](https://tailwindcss.com/)
[![Prisma](https://img.shields.io/badge/Prisma-5.22-2D3748.svg)](https://www.prisma.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791.svg)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D.svg)](https://redis.io/)
[![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4.svg)](https://cloud.google.com/)
[![Terraform](https://img.shields.io/badge/Terraform-7B42BC.svg)](https://www.terraform.io/)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Velox Web Frontend](#velox-web-frontend)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Infrastructure](#infrastructure)
- [API Reference](#api-reference)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

Velox is a production-ready AI voice agent platform designed for enterprise applications. It enables organizations to create and manage intelligent AI agents capable of handling voice calls and text-based conversations with advanced capabilities including:

- 🎙️ **Voice Interactions** - Integration with ElevenLabs and Deepgram for natural voice conversations
- 💬 **Chat Support** - Text-based conversations with persistent context
- 🛠️ **Tool Integration** - Extensible tool system for agents to perform actions (check orders, book tickets, etc.)
- 📊 **Analytics Dashboard** - Monitor conversation costs, sentiment, and performance
- 🔐 **Multi-Tenancy** - Secure organization-level isolation with role-based access control
- 🎨 **Visual Flow Builder** - Intuitive drag-and-drop interface for designing AI agent conversation flows

---

## 🎨 Velox Web Frontend

The Velox Web Frontend is a modern React 19 application built with Vite and TypeScript, featuring a visual flow builder for designing AI agent conversation flows.

### Tech Stack

| Technology | Purpose |
|------------|---------|
| React 19 | UI framework |
| Vite 7 | Build tool and dev server |
| TypeScript 5.9 | Type safety |
| Tailwind CSS 4 | Styling |
| @xyflow/react | Flow-based diagram library |
| React Router 7 | Client-side routing |
| TanStack Query | Data fetching and caching |
| Zustand | State management |
| Axios | HTTP client |
| Sonner | Toast notifications |
| Radix UI | Accessible UI primitives |

### Key Features

#### Visual Flow Builder
The flow builder provides a drag-and-drop interface for designing AI agent conversation flows with the following node types:

| Node Type | Description | Properties |
|-----------|-------------|------------|
| **Start Node** (Green) | Entry point of the conversation | Label, Greeting message |
| **Prompt Node** (Blue) | LLM prompt configuration | Label, System prompt, Temperature, Max tokens |
| **Tool Node** (Orange) | Tool/action execution | Label, Tool name, Tool config (JSON) |
| **Handoff Node** (Purple) | Transfer to human agent | Label, Target (phone/agent ID), Reason |
| **Condition Node** (Yellow) | Branching logic | Label, Condition expression, True/False labels |
| **End Node** (Red) | Conversation termination | Label, Farewell message |

#### UI Components
A comprehensive set of reusable UI components built with Radix UI and Tailwind CSS:

- **Button** - Multiple variants (default, outline, secondary, ghost)
- **Card** - Content containers with header, content, and title
- **Input** - Text input fields
- **Textarea** - Multi-line text input
- **Label** - Form labels with Radix UI integration
- **Select** - Dropdown select with search capability
- **Badge** - Status indicators and labels

### Project Structure

```
velox-web/
├── src/
│   ├── components/
│   │   ├── flow/                  # Flow builder components
│   │   │   ├── FlowEditor.tsx     # Main flow editor with React Flow
│   │   │   ├── FlowToolbar.tsx    # Toolbar for adding nodes
│   │   │   ├── NodePropertiesPanel.tsx  # Properties panel for selected node
│   │   │   └── nodes/             # Custom node components
│   │   │       ├── index.ts       # Node exports

│   │   │       ├── PromptNode.tsx # Prompt node (blue)
│   │   │       ├── ToolNode.tsx   # Tool node (orange)
│   │   │       ├── HandoffNode.tsx # Handoff node (purple)
│   │   │       ├── ConditionNode.tsx # Condition node (yellow)
│   │   │       └── EndNode.tsx    # End node (red)
│   │   │
│   │   └── ui/                    # Reusable UI components
│   │       ├── button.tsx
│   │       ├── card.tsx
│   │       ├── input.tsx
│   │       ├── textarea.tsx
│   │       ├── label.tsx
│   │       ├── select.tsx
│   │       └── badge.tsx
│   │
│   ├── pages/                     # Page components
│   │   └── AgentFlowBuilder.tsx   # Agent flow builder page
│   │
│   ├── types/                     # TypeScript types
│   │   └── flow.ts                # Flow-related type definitions
│   │
│   ├── lib/                       # Utilities
│   │   ├── api.ts                 # Axios API client
│   │   └── utils.ts               # Utility functions
│   │
│   ├── App.tsx                    # Main app component
│   ├── main.tsx                   # Entry point
│   └── index.css                  # Global styles
│
├── package.json
├── tsconfig.json
├── tsconfig.app.json
├── tsconfig.node.json
├── vite.config.ts
└── tailwind.config.js
```

### Flow Data Structure

```typescript
interface AgentFlow {
  nodes: Array<{
    id: string
    type: string
    position: { x: number; y: number }
    data: Record<string, unknown>
  }>
```
    id: string
    source: string
    target: string
    label?: string
  }>
}

interface StartNodeData {
  label: string
  greeting?: string
}

interface PromptNodeData {
  label: string
  systemPrompt: string
  temperature?: number
  maxTokens?: number
}

interface ToolNodeData {
  label: string
  toolName: string
  toolConfig: Record<string, unknown>
}

interface HandoffNodeData {
  label: string
  target: string
  reason?: string
}

interface ConditionNodeData {
  label: string
  condition: string
  trueLabel?: string
  falseLabel?: string
}

interface EndNodeData {
  label: string
  farewell?: string
}
```

### Getting Started

```bash
# Navigate to velox-web directory
cd velox-web

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Environment Variables

```env
VITE_API_URL=http://localhost:3000
```

### API Integration

The frontend uses an Axios-based API client with:

- **Request interceptor** - Automatically adds JWT token from localStorage
- **Response interceptor** - Handles 401 errors by redirecting to login
- **Configurable base URL** - Via `VITE_API_URL` environment variable

### Dependencies

```json
{
  "@xyflow/react": "^12.10.0",
  "@radix-ui/react-*": "^1.1.0",
  "@tanstack/react-query": "^5.90.0",
  "react-router-dom": "^7.13.0",
  "zustand": "^5.0.10",
  "axios": "^1.13.0",
  "sonner": "^2.0.0",
  "lucide-react": "^0.563.0"
}
```

---

## 🏗️ Architecture
┌─────────────────────────────────────────────────────────────────────────┐
│                           Velox AI Platform                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐ │
│  │   Frontend  │◄──►│  velox-api  │◄──►│      Google Cloud           │ │
│  │   (React)   │    │  (Node.js)  │    │                             │ │
│  └─────────────┘    └─────────────┘    │  ┌───────────────────────┐  │ │
│                                        │  │   Cloud SQL (Postgres) │  │ │
│  ┌─────────────┐    ┌─────────────┐    │  └───────────────────────┘  │ │
│  │   Twilio    │◄──►│  velox-api  │    │  ┌───────────────────────┐  │ │
│  │   (Voice)   │    │  (Node.js)  │    │  │   Redis Cache         │  │ │
│  └─────────────┘    └─────────────┘    │  └───────────────────────┘  │ │
│                                        │  ┌───────────────────────┐  │ │
│  ┌─────────────┐    ┌─────────────┐    │  │   AI Platform         │  │ │
│  │ ElevenLabs/ │◄──►│  velox-api  │    │  │   (Gemini, etc.)      │  │ │
│  │  Deepgram   │    │  (Node.js)  │    │  └───────────────────────┘  │ │
│  └─────────────┘    └─────────────┘    │                             │ │
│                                        └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Components

1. **velox-api** - TypeScript/Express API server with Prisma ORM
2. **Infrastructure** - Terraform configurations for Google Cloud Platform
3. **Database** - PostgreSQL 15 with pgvector for vector storage
4. **Cache** - Redis 7 for session management and caching

---

## ✨ Features

### Multi-Tenant Architecture
- **Organizations** - Complete tenant isolation with unique API keys
- **Users** - Role-based access control (ADMIN, EDITOR, VIEWER)
- **Credit System** - Track usage and manage billing

### AI Agent Management
- **Custom System Prompts** - Configure agent personality and behavior
- **Voice Integration** - Support for ElevenLabs and Deepgram voices
- **Tool System** - Extensible tools for agent actions (check orders, stock lookup)
- **LLM Configuration** - Adjust model parameters (temperature, model selection)

### Execution Loop: "Thinking... Action... Speaking."
- **Intent Detection** - Gemini configured with function calling to detect when tools are needed
- **Tool Execution** - Seamless tool execution with result injection back to Gemini
- **RAG Integration** - Knowledge base search before LLM response generation
- **Real-time Pipeline** - Ear → Brain → Mouth flow with WebSocket audio streaming

### Conversation Intelligence
- **Real-time Monitoring** - Track active conversations
- **Cost Tracking** - Accrue and monitor costs per conversation
- **Sentiment Analysis** - Automatic sentiment scoring (-1.0 to 1.0)
- **Audit Trail** - Complete message history with token usage

### Developer Experience
- **TypeScript** - Full type safety across the codebase
- **Prisma ORM** - Type-safe database access
- **Docker Support** - Containerized development environment
- **Terraform IaC** - Infrastructure as Code for reproducible deployments

---

## 📁 Project Structure

```
Velox_AI/
├── README.md                      # This file
├── docker-compose.yml             # Local development environment (PostgreSQL + Redis)
├── cloudbuild.yaml                # Google Cloud Build CI/CD pipeline
│
├── velox-api/                     # Main API application (TypeScript/Express)
│   ├── package.json               # Node.js dependencies
│   ├── tsconfig.json              # TypeScript configuration
│   ├── Dockerfile                 # Multi-stage Docker build
│   ├── prisma.config.ts           # Prisma configuration
│   ├── simulate-twilio.js         # Twilio simulation script for local testing
│   │
│   ├── prisma/                    # Database layer
│   │   ├── schema.prisma          # Database schema (Organizations, Users, Agents, Conversations, Messages)
│   │   ├── seed.ts                # Database seeding script
│   │   └── migrations/            # Database migrations
│   │
│   └── src/                       # Application source code
│       ├── server.ts              # Application entry point + WebSocket server setup
│       ├── app.ts                 # Express app configuration (middleware, health checks)
│       │
│       ├── config/
│       │   └── redis.ts           # Redis client configuration
│       │
│       ├── middleware/
│       │   ├── rateLimiter.ts     # Redis-based rate limiting middleware
│       │   └── twilioAuth.ts      # Twilio webhook signature validation
│       │
│       ├── routes/
│       │   └── voice.ts           # Twilio voice webhook + TwiML response
│       │
│       ├── services/
│       │   ├── sessionService.ts  # Redis session state management (CallStage enum)
│       │   └── transcriptionService.ts  # Deepgram transcription integration (Nova-2)
│       │
│       └── websocket/
│           └── streamHandler.ts   # WebSocket handler for real-time audio streams
│
└── infrastructure/                # Terraform Infrastructure as Code (Google Cloud)
    ├── .terraform.lock.hcl        # Terraform provider lock file
    ├── main.tf                    # Main infrastructure definition (VPC, Cloud SQL, Redis)
    ├── variables.tf               # Variable declarations
    ├── provider.tf                # GCP provider configuration
    ├── outputs.tf                 # Output values (database IP, Redis host)
    └── terraform.tfstate          # Terraform state file
```

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Node.js 18+
- Google Cloud CLI (for production deployment)
- Terraform 1.0+

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/velox-ai.git
   cd velox-ai
   ```

2. **Start the infrastructure**
   ```bash
   docker-compose up -d
   ```

3. **Set up environment variables**
   ```bash
   cp velox-api/.env.example velox-api/.env
   # Edit .env with your configuration
   ```

4. **Initialize the database**
   ```bash
   cd velox-api
   npx prisma migrate dev
   npx prisma db seed
   ```

5. **Start the API server**
   ```bash
   npm run dev
   ```

6. **Access the application**
   - API: http://localhost:3000
   - Database: localhost:5432
   - Redis: localhost:6379

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the `velox-api` directory:

```env
# Database
DATABASE_URL="postgresql://postgres:devpass@localhost:5432/velox_local"

# Redis
REDIS_URL="redis://localhost:6379"

# Application
NODE_ENV="development"
PORT=3000

# API Keys
ELEVENLABS_API_KEY="your-elevenlabs-key"
DEEPGRAM_API_KEY="your-deepgram-key"
TWILIO_ACCOUNT_SID="your-twilio-sid"
TWILIO_AUTH_TOKEN="your-twilio-token"
GOOGLE_AI_API_KEY="your-google-ai-key"

# JWT
JWT_SECRET="your-jwt-secret-key"
```

### Terraform Variables

Configure `infrastructure/terraform.tfvars`:

```hcl
project_id = "velox-ai-prod-2025"
region     = "europe-west2"
```

---

## ☁️ Infrastructure

### Google Cloud Resources

The infrastructure is provisioned using Terraform and includes:

| Resource | Type | Description |
|----------|------|-------------|
| VPC Network | `google_compute_network` | Isolated network for all resources |
| Cloud SQL | `google_sql_database_instance` | PostgreSQL 15 with private networking |
| Redis | `google_redis_instance` | Managed Redis cache (5GB) |
| Service Networking | `google_service_networking_connection` | Private service connectivity |

### Deployment Steps

1. **Initialize Terraform**
   ```bash
   cd infrastructure
   terraform init
   ```

2. **Plan deployment**
   ```bash
   terraform plan -out=tfplan
   ```

3. **Apply changes**
   ```bash
   terraform apply tfplan
   ```

4. **Retrieve outputs**
   ```bash
   terraform output
   ```

---

## 📚 API Reference

### Authentication

All API requests require authentication via API key:

```http
Headers:
  Authorization: Bearer <org_api_key>
  X-API-Key: <org_api_key>
```

### Endpoints

#### Organizations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/organizations` | Create new organization |
| GET | `/api/organizations/:id` | Get organization details |
| PUT | `/api/organizations/:id` | Update organization |
| DELETE | `/api/organizations/:id` | Delete organization |

#### Users

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/users` | Create new user |
| GET | `/api/users` | List organization users |
| GET | `/api/users/:id` | Get user details |
| PUT | `/api/users/:id` | Update user |
| DELETE | `/api/users/:id` | Delete user |

#### Agents

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/agents` | Create new agent |
| GET | `/api/agents` | List organization agents |
| GET | `/api/agents/:id` | Get agent details |
| PUT | `/api/agents/:id` | Update agent |
| DELETE | `/api/agents/:id` | Delete agent |

#### Conversations

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/conversations` | List conversations |
| GET | `/api/conversations/:id` | Get conversation details |
| GET | `/api/conversations/:id/messages` | Get conversation messages |
| PUT | `/api/conversations/:id` | Update conversation status |

---

## 💻 Development

### Database Schema

The application uses a multi-tenant schema with the following models:

```prisma
model Organization {
  id             String  @id @default(uuid())
  name           String
  slug           String  @unique
  stripe_id      String?
  credit_balance Int     @default(0)
  api_key_hash   String  @unique
  
  users      User[]
  agents     Agent[]
  created_at DateTime @default(now())
}

model User {
  id       String @id @default(uuid())
  email    String @unique
  password String
  role     Role   @default(VIEWER)
  org_id   String
  
  org Organization @relation(fields: [org_id], references: [id])
}

model Agent {
  id            String @id @default(uuid())
  name          String
  system_prompt String
  voice_id      String
  tools_enabled Json   @default("[]")
  llm_config    Json   @default("{}")
  org_id        String
  
  org           Organization  @relation(fields: [org_id], references: [id])
  conversations Conversation[]
}

model Conversation {
  id             String             @id @default(uuid())
  twilio_sid     String             @unique
  status         ConversationStatus @default(ACTIVE)
  cost_accrued   Float              @default(0.000)
  sentiment_score Float?
  agent_id       String
  
  agent    Agent      @relation(fields: [agent_id], references: [id])
  messages Message[]
}

model Message {
  id      String @id @default(uuid())
  role    String
  content String
  tokens  Int    @default(0)
  latency_ms Int  @default(0)
  conversation_id String
  
  conversation Conversation @relation(fields: [conversation_id], references: [id])
}
```

### Running Tests

```bash
cd velox-api
npm test
```

### Database Migrations

```bash
# Create new migration
npx prisma migrate dev --name <migration_name>

# Apply migrations in production
npx prisma migrate deploy

# Reset database (development only)
npx prisma migrate reset
```

---

## 🚢 Deployment

### Production Checklist

- [ ] Set `NODE_ENV=production`
- [ ] Configure production database credentials
- [ ] Enable SSL/TLS
- [ ] Set up monitoring and alerting
- [ ] Configure backup and disaster recovery
- [ ] Review and apply security hardening

### Docker Production

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Terraform Production

```bash
cd infrastructure
terraform workspace new prod
terraform plan -var="environment=prod"
terraform apply
```

---

## 🎯 Execution Loop: "Thinking... Action... Speaking."

The Velox AI voice agent follows a sophisticated execution loop designed for natural voice conversations. This loop orchestrates the interaction between the user, the AI brain (LLM), and tool execution, ensuring seamless conversation flow.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  EXECUTION LOOP: "Thinking... Action... Speaking."       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐  │
│  │   USER      │────►│      AI BRAIN    │────►│      SPEAKING       │  │
│  │   SPEAKS    │     │   (Thinking)     │     │    (TTS Output)     │  │
│  └─────────────┘     └──────────────────┘     └─────────────────────┘  │
│        │                      │                        ▲                │
│        │                      │                        │                │
│        ▼                      ▼                        │                │
│  ┌─────────────┐     ┌──────────────────┐              │                │
│  │   SPEECH    │────►│   TOOL ACTION    │──────────────┘                │
│  │  TO TEXT    │     │   (Database/API) │                              │
│  └─────────────┘     └──────────────────┘                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### How It Works

#### 1. Intent Detection & Tool Configuration

Gemini is configured with function calling capability. When the user speaks, their audio is transcribed to text and sent to Gemini along with available tool definitions.

```typescript
// llmService.ts - Tool configuration
config: {
  systemInstruction: instructions,
  tools: [{ functionDeclarations: tools }],
}
```

#### 2. Tool Execution Loop

When Gemini detects the need for a tool (e.g., checking order status or stock availability), it returns `functionCalls` instead of text. The system:

1. **Detects tool intent** - `let functionCalls = response.functionCalls`
2. **Executes the tool** - Calls the appropriate function from the tool registry
3. **Injects the result** - Feeds the JSON result back to Gemini as `functionResponse`
4. **Generates final response** - Gemini produces the natural language answer

```typescript
// Execute tool
const apiResult = await functionToCall(args);

// Feed result back to Gemini
response = await client.models.generateContent({
  model: this.modelName,
  contents: [{
    role: "tool",
    parts: [{
      functionResponse: {
        name: toolName,
        response: apiResult,
      },
    }],
  }],
});

// Final output
const finalText = response.text;
```

#### 3. Text-to-Speech Output

The final response is converted to audio using Deepgram's Aura voice engine and streamed back to the user in mulaw 8000Hz format (Twilio compatible).

### Implemented Features

| Feature | Status | Description |
|---------|--------|-------------|
| Tool Configuration | ✅ | Gemini configured with functionDeclarations |
| Intent Detection | ✅ | functionCalls detection in LLMService |
| Tool Execution Loop | ✅ | While loop handles sequential tool calls |
| Result Injection | ✅ | functionResponse sent back to Gemini |
| TTS Output | ✅ | Deepgram Aura voice at 8000Hz mulaw |

### Available Tools

Agents can invoke these tools during conversation:

- **check_order_status** - Look up the current status of a customer's order using their Order ID
- **check_item_stock** - Check if an item is available in the warehouse

Each tool is registered in the tool registry and can be extended with additional functionality.

### Processing Flow

```
User Audio → Deepgram STT → Gemini (with tools) → Tool Execution → Result Injection → Gemini Response → TTS → User
```

The entire pipeline is managed by the `CallOrchestrator` which handles:
- WebSocket audio streaming
- Real-time transcription
- LLM response generation
- TTS audio playback
- Interruption handling (barge-in)
- Metrics tracking



---

## 📈 Monitoring

### Key Metrics

- **Conversation Costs** - Track cost_accrued per conversation
- **Latency** - Monitor latency_ms for performance
- **Sentiment** - Analyze sentiment_score trends
- **Token Usage** - Monitor tokens per message

### Health Checks

```bash
# API health
curl http://localhost:3000/health

# Database health
curl http://localhost:3000/health/db

# Redis health
curl http://localhost:3000/health/redis
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

