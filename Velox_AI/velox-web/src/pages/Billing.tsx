// src/pages/Billing.tsx

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Check, Loader2, TrendingUp, TrendingDown, CreditCard, DollarSign, Clock, Zap, ArrowLeft } from 'lucide-react'
import { toast } from 'sonner'
import { Link } from 'react-router-dom'
import api from '@/lib/api'

// Animation variants
const fadeInUp = {
  hidden: { opacity: 0, y: 40 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.5, ease: 'easeOut' as const }
  }
}

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15,
      delayChildren: 0.1
    }
  }
}

const scaleIn = {
  hidden: { opacity: 0, scale: 0.9 },
  visible: { 
    opacity: 1, 
    scale: 1,
    transition: { duration: 0.4, ease: 'easeOut' as const }
  }
}

const slideInRight = {
  hidden: { opacity: 0, x: 40 },
  visible: { 
    opacity: 1, 
    x: 0,
    transition: { duration: 0.5, ease: 'easeOut' as const }
  }
}

interface Plan {
  name: string
  type: 'STARTER' | 'PRO' | 'ENTERPRISE'
  price: number
  minutes: number
  features: string[]
  popular?: boolean
}

const plans: Plan[] = [
  {
    name: 'Starter',
    type: 'STARTER',
    price: 49,
    minutes: 1000,
    features: [
      '1,000 minutes per month',
      'Up to 5 agents',
      'Basic analytics',
      'Email support',
      'Knowledge base (1 GB)',
    ],
  },
  {
    name: 'Pro',
    type: 'PRO',
    price: 199,
    minutes: 5000,
    popular: true,
    features: [
      '5,000 minutes per month',
      'Unlimited agents',
      'Advanced analytics',
      'Priority support',
      'Knowledge base (10 GB)',
      'Custom voice training',
      'Webhook integrations',
    ],
  },
  {
    name: 'Enterprise',
    type: 'ENTERPRISE',
    price: 499,
    minutes: 20000,
    features: [
      '20,000 minutes per month',
      'Unlimited agents',
      'Real-time analytics',
      '24/7 dedicated support',
      'Unlimited knowledge base',
      'Custom voice training',
      'Advanced integrations',
      'SLA guarantee',
      'Custom contracts',
    ],
  },
]

export default function Billing() {
  const [loading, setLoading] = useState(false)
  const [billingInfo, setBillingInfo] = useState<any>(null)
  const [transactions, setTransactions] = useState<any[]>([])
  const orgId = 'test-org-id' // TODO: Get from auth context

  useEffect(() => {
    loadBillingInfo()
  }, [])

  const loadBillingInfo = async () => {
    try {
      const response = await api.get(`/api/billing/${orgId}`)
      setBillingInfo(response.data)
      setTransactions(response.data.transactions || [])
    } catch (error) {
      console.error('Failed to load billing info:', error)
    }
  }

  const handleSubscribe = async (planType: string) => {
    setLoading(true)
    try {
      const response = await api.post('/api/billing/checkout', {
        orgId,
        planType,
        successUrl: `${window.location.origin}/billing?success=true`,
        cancelUrl: `${window.location.origin}/billing?canceled=true`,
      })

      if (response.data.sessionId) {
        window.location.href = `${import.meta.env.VITE_STRIPE_CHECKOUT_URL}?session_id=${response.data.sessionId}`
      }
    } catch (error: any) {
      console.error('Checkout failed:', error)
      toast.error('Failed to start checkout')
    } finally {
      setLoading(false)
    }
  }

  const handleCancelSubscription = async () => {
    if (!confirm('Are you sure you want to cancel your subscription?')) {
      return
    }

    try {
      await api.post(`/api/billing/${orgId}/cancel`)
      toast.success('Subscription cancelled')
      loadBillingInfo()
    } catch (error) {
      console.error('Failed to cancel:', error)
      toast.error('Failed to cancel subscription')
    }
  }

  const formatMinutes = (minutes: number) => {
    const hours = Math.floor(minutes / 60)
    const mins = minutes % 60
    return hours > 0 ? `${hours}h ${mins}m` : `${mins}m`
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header with back button */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="border-b bg-background/95 backdrop-blur"
      >
        <div className="container mx-auto px-4 py-4">
          <Button variant="ghost" size="sm" asChild>
            <Link to="/" className="gap-2">
              <ArrowLeft className="h-4 w-4" />
              Back to Home
            </Link>
          </Button>
        </div>
      </motion.div>

      <div className="container mx-auto p-8 max-w-7xl">
        <motion.div
          initial="hidden"
          animate="visible"
          variants={staggerContainer}
          className="space-y-12"
        >
          {/* Hero Section */}
          <motion.div variants={fadeInUp} className="text-center space-y-4">
            <h1 className="text-5xl font-bold bg-gradient-to-r from-primary to-primary/50 bg-clip-text text-transparent">
              Choose Your Plan
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Flexible pricing that scales with your business. Start free, upgrade anytime.
            </p>
          </motion.div>

          {/* Current Balance Card with Animation */}
          <motion.div variants={scaleIn}>
            <Card className="border-2 border-primary bg-gradient-to-br from-primary/5 via-background to-background overflow-hidden relative">
              {/* Animated background gradient */}
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-primary/10 to-transparent"
                animate={{
                  x: ['-100%', '100%'],
                }}
                transition={{
                  duration: 3,
                  repeat: Infinity,
                  ease: 'linear',
                }}
              />
              
              <CardHeader className="relative">
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-2xl flex items-center gap-2">
                      <CreditCard className="h-6 w-6 text-primary" />
                      Current Balance
                    </CardTitle>
                    <CardDescription className="mt-2">
                      Minutes available for AI calls
                    </CardDescription>
                  </div>
                  {billingInfo?.current_plan && (
                    <Badge variant="secondary" className="text-base px-4 py-2">
                      {billingInfo.current_plan}
                    </Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent className="relative">
                <div className="grid md:grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <div className="text-5xl font-bold text-primary">
                      {formatMinutes(billingInfo?.credit_balance || 0)}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {billingInfo?.credit_balance || 0} minutes remaining
                    </div>
                  </div>
                  
                  {billingInfo?.subscription && (
                    <>
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-muted-foreground">
                          <Clock className="h-4 w-4" />
                          <span className="text-sm">Next Renewal</span>
                        </div>
                        <div className="text-xl font-semibold">
                          {new Date(
                            billingInfo.subscription.current_period_end * 1000
                          ).toLocaleDateString()}
                        </div>
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-muted-foreground">
                          <DollarSign className="h-4 w-4" />
                          <span className="text-sm">Monthly Cost</span>
                        </div>
                        <div className="text-xl font-semibold">
                          ${plans.find(p => p.type === billingInfo.current_plan)?.price || 0}
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Plans Grid */}
          <motion.div variants={fadeInUp}>
            <div className="grid md:grid-cols-3 gap-8">
              {plans.map((plan, index) => {
                const isCurrentPlan = billingInfo?.current_plan === plan.type
                return (
                  <motion.div
                    key={plan.type}
                    variants={scaleIn}
                    custom={index}
                    whileHover={{ 
                      scale: 1.03,
                      transition: { duration: 0.2 }
                    }}
                  >
                    <Card
                      className={`h-full relative overflow-hidden ${
                        plan.popular 
                          ? 'border-2 border-primary shadow-xl shadow-primary/20' 
                          : isCurrentPlan 
                          ? 'border-2 border-primary' 
                          : ''
                      }`}
                    >
                      {plan.popular && (
                        <div className="absolute top-0 right-0">
                          <div className="bg-primary text-primary-foreground px-4 py-1 text-xs font-semibold">
                            MOST POPULAR
                          </div>
                        </div>
                      )}
                      
                      <CardHeader className="pb-8">
                        <div className="flex items-center justify-between mb-4">
                          <CardTitle className="text-2xl">{plan.name}</CardTitle>
                          {isCurrentPlan && (
                            <Badge variant="outline" className="border-primary">
                              Current
                            </Badge>
                          )}
                        </div>
                        <div className="space-y-1">
                          <div className="flex items-baseline gap-1">
                            <span className="text-5xl font-bold">${plan.price}</span>
                            <span className="text-muted-foreground">/month</span>
                          </div>
                          <CardDescription className="text-base">
                            {plan.minutes.toLocaleString()} minutes included
                          </CardDescription>
                        </div>
                      </CardHeader>
                      
                      <CardContent className="space-y-6">
                        <ul className="space-y-3">
                          {plan.features.map((feature, i) => (
                            <motion.li
                              key={i}
                              initial={{ opacity: 0, x: -20 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: i * 0.05 }}
                              className="flex items-start gap-3 text-sm"
                            >
                              <div className="mt-0.5">
                                <Check className="h-5 w-5 text-primary shrink-0" />
                              </div>
                              <span>{feature}</span>
                            </motion.li>
                          ))}
                        </ul>
                        
                        <Button
                          className="w-full h-11"
                          onClick={() => handleSubscribe(plan.type)}
                          disabled={loading || isCurrentPlan}
                          variant={plan.popular ? 'default' : isCurrentPlan ? 'outline' : 'outline'}
                        >
                          {loading ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : isCurrentPlan ? (
                            'Current Plan'
                          ) : (
                            <>
                              <Zap className="h-4 w-4 mr-2" />
                              Subscribe Now
                            </>
                          )}
                        </Button>
                      </CardContent>
                    </Card>
                  </motion.div>
                )
              })}
            </div>
          </motion.div>

          {/* Transaction History */}
          <motion.div variants={fadeInUp}>
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl">Transaction History</CardTitle>
                <CardDescription>Recent credits and usage</CardDescription>
              </CardHeader>
              <CardContent>
                {transactions.length === 0 ? (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="text-center py-12"
                  >
                    <CreditCard className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
                    <p className="text-muted-foreground">No transactions yet</p>
                    <p className="text-sm text-muted-foreground mt-2">
                      Your transaction history will appear here
                    </p>
                  </motion.div>
                ) : (
                  <motion.div
                    variants={staggerContainer}
                    initial="hidden"
                    animate="visible"
                    className="space-y-3"
                  >
                    {transactions.map((transaction, index) => (
                      <motion.div
                        key={transaction.id}
                        variants={slideInRight}
                        custom={index}
                        whileHover={{ scale: 1.02 }}
                        className="flex items-center justify-between p-4 border rounded-lg hover:shadow-md transition-shadow"
                      >
                        <div className="flex items-center gap-4">
                          <motion.div
                            whileHover={{ rotate: 360 }}
                            transition={{ duration: 0.5 }}
                          >
                            {transaction.type === 'CREDIT' ? (
                              <div className="h-12 w-12 rounded-full bg-green-100 dark:bg-green-900 flex items-center justify-center">
                                <TrendingUp className="h-6 w-6 text-green-600 dark:text-green-400" />
                              </div>
                            ) : (
                              <div className="h-12 w-12 rounded-full bg-red-100 dark:bg-red-900 flex items-center justify-center">
                                <TrendingDown className="h-6 w-6 text-red-600 dark:text-red-400" />
                              </div>
                            )}
                          </motion.div>
                          <div>
                            <p className="font-medium">
                              {transaction.description}
                            </p>
                            <p className="text-sm text-muted-foreground">
                              {new Date(transaction.created_at).toLocaleString()}
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p
                            className={`text-lg font-bold ${
                              transaction.type === 'CREDIT'
                                ? 'text-green-600 dark:text-green-400'
                                : 'text-red-600 dark:text-red-400'
                            }`}
                          >
                            {transaction.type === 'CREDIT' ? '+' : '-'}
                            {formatMinutes(transaction.amount)}
                          </p>
                          <p className="text-sm text-muted-foreground">
                            Balance: {formatMinutes(transaction.balance_after)}
                          </p>
                        </div>
                      </motion.div>
                    ))}
                  </motion.div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* Cancel Subscription */}
          {billingInfo?.subscription && (
            <motion.div variants={fadeInUp}>
              <Card className="border-2 border-destructive/50 bg-destructive/5">
                <CardHeader>
                  <CardTitle className="text-destructive flex items-center gap-2">
                    ⚠️ Danger Zone
                  </CardTitle>
                  <CardDescription>
                    Once you cancel, you'll lose access to all premium features at the end of your billing period
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Button
                    variant="destructive"
                    onClick={handleCancelSubscription}
                    className="w-full md:w-auto"
                  >
                    Cancel Subscription
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </motion.div>
      </div>
    </div>
  )
}