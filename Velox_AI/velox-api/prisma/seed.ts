import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

async function main() {
  // 1. Create a Tenant
  const org = await prisma.organization.create({
    data: {
      name: "Velox Test Corp",
      slug: "velox-corp",
      api_key_hash: "secret_hash_123",
      credit_balance: 5000
    }
  })

  // 2. Create an Agent for that Tenant
  const agent = await prisma.agent.create({
    data: {
      name: "Support Bot",
      system_prompt: "You are a helpful assistant.",
      voice_id: "voice_abc_123",
      org_id: org.id,
      llm_config: { model: "gemini-1.5-flash", temp: 0.5 }
    }
  })

  console.log(`✅ Seeded: Org ${org.name} | Agent ${agent.name}`)
}

main()
  .then(async () => {
    await prisma.$disconnect()
  })
  .catch(async (e) => {
    console.error(e)
    await prisma.$disconnect()
    process.exit(1)
  })