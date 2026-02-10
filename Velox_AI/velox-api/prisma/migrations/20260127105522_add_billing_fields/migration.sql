/*
  Warnings:

  - You are about to drop the column `stripe_id` on the `organizations` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "organizations" DROP COLUMN "stripe_id",
ADD COLUMN     "billing_email" TEXT,
ADD COLUMN     "current_plan" TEXT,
ADD COLUMN     "stripe_customer_id" TEXT,
ADD COLUMN     "stripe_subscription_id" TEXT;

-- CreateTable
CREATE TABLE "transactions" (
    "id" TEXT NOT NULL,
    "org_id" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "amount" INTEGER NOT NULL,
    "description" TEXT NOT NULL,
    "balance_after" INTEGER NOT NULL,
    "conversation_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "transactions_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "transactions_org_id_idx" ON "transactions"("org_id");

-- CreateIndex
CREATE INDEX "transactions_created_at_idx" ON "transactions"("created_at");

-- AddForeignKey
ALTER TABLE "transactions" ADD CONSTRAINT "transactions_org_id_fkey" FOREIGN KEY ("org_id") REFERENCES "organizations"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
