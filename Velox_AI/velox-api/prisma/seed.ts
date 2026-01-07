import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

async function main() {
  console.log(' Starting seed...')

  // 1. Upsert Organization (Create if not exists, otherwise update)
  const org = await prisma.organization.upsert({
    where: { slug: 'velox-corp' }, // Check if this slug exists
    update: {}, // If it exists, do nothing (or update fields if you want)
    create: {
      name: "Velox Test Corp",
      slug: "velox-corp",
      api_key_hash: "secret_hash_123",
      credit_balance: 5000
    }
  })

  // 2. Upsert Agent
  // We first need to find if the agent exists to avoid duplicates, 
  // but since Agent doesn't have a unique slug in our schema, 
  // we'll just delete existing agents for this org to keep it clean.
  await prisma.agent.deleteMany({ where: { org_id: org.id } })

  const agent = await prisma.agent.create({
    data: {
      name: "Support Bot",
      system_prompt: "You are a helpful assistant.",
      voice_id: "voice_abc_123",
      org_id: org.id,
      llm_config: { model: "gemini-1.5-flash", temp: 0.5 }
    }
  })

  console.log(` Seeded: Org "${org.name}" | Agent "${agent.name}"`)
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