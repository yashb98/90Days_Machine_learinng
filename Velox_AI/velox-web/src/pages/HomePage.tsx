// src/pages/HomePage.tsx

import { motion } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Link } from 'react-router-dom'
import {
  Bot,
  Zap,
  Shield,
  TrendingUp,
  MessageSquare,
  PhoneCall,
  BarChart3,
  Sparkles,
  ArrowRight,
  CheckCircle2,
  Users,
  Clock,
  DollarSign,
} from 'lucide-react'

// Animation variants
const fadeInUp = {
  hidden: { opacity: 0, y: 60 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.6, ease: 'easeOut' as const }
  }
}

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.2
    }
  }
}

const scaleIn = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: { 
    opacity: 1, 
    scale: 1,
    transition: { duration: 0.5, ease: 'easeOut' as const }
  }
}

const slideInLeft = {
  hidden: { opacity: 0, x: -60 },
  visible: { 
    opacity: 1, 
    x: 0,
    transition: { duration: 0.6, ease: 'easeOut' as const }
  }
}

const slideInRight = {
  hidden: { opacity: 0, x: 60 },
  visible: { 
    opacity: 1, 
    x: 0,
    transition: { duration: 0.6, ease: 'easeOut' as const }
  }
}

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <motion.nav 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
      >
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bot className="h-8 w-8 text-primary" />
            <span className="text-xl font-bold text-foreground dark:text-white">Velox AI</span>
          </div>
          <div className="flex items-center gap-4">
            <Button variant="ghost" asChild>
              <Link to="/billing" className="text-foreground dark:text-gray-200">Pricing</Link>
            </Button>
            <Button variant="ghost" className="text-foreground dark:text-gray-200">Documentation</Button>
            <Button asChild>
              <Link to="/agents/demo/playground">Try Demo</Link>
            </Button>
          </div>
        </div>
      </motion.nav>

      {/* Hero Section */}
      <section className="relative overflow-hidden py-20 md:py-32">
        {/* Background gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-background to-background" />
        
        <div className="container mx-auto px-4 relative">
          <motion.div
            initial="hidden"
            animate="visible"
            variants={staggerContainer}
            className="max-w-4xl mx-auto text-center space-y-8"
          >
            <motion.div variants={fadeInUp}>
              <Badge variant="secondary" className="mb-4">
                <Sparkles className="h-3 w-3 mr-1" />
                Powered by Gemini 2.5
              </Badge>
            </motion.div>

            <motion.h1 
              variants={fadeInUp}
              className="text-5xl md:text-7xl font-bold tracking-tight text-foreground"
            >
              Build AI Voice Agents
              <span className="block text-primary mt-2">In Minutes, Not Months</span>
            </motion.h1>

            <motion.p 
              variants={fadeInUp}
              className="text-xl text-muted-foreground max-w-2xl mx-auto"
            >
              The no-code platform for creating intelligent voice AI agents. 
              Handle customer calls, appointments, and support with advanced RAG and tool integration.
            </motion.p>

            <motion.div 
              variants={fadeInUp}
              className="flex gap-4 justify-center"
            >
              <Button size="lg" className="text-lg" asChild>
                <Link to="/agents/demo/flow">
                  Start Building Free
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Link>
              </Button>
              <Button size="lg" variant="outline" className="text-orange-400 text-lg" asChild>
                <Link to="/agents/demo/playground">
                  Try Playground
                </Link>
              </Button>
            </motion.div>

            <motion.div 
              variants={fadeInUp}
              className="flex items-center justify-center gap-8 text-sm text-muted-foreground"
            >
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                No credit card required
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                1,000 free minutes
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                Setup in 5 minutes
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 border-y bg-muted/50">
        <div className="container mx-auto px-4">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="grid md:grid-cols-4 gap-8"
          >
            {[
              { icon: Users, label: 'Active Users', value: '10K+' },
              { icon: PhoneCall, label: 'Calls Handled', value: '1M+' },
              { icon: Clock, label: 'Avg Response', value: '< 2s' },
              { icon: TrendingUp, label: 'Uptime', value: '99.9%' },
            ].map((stat, i) => (
              <motion.div key={i} variants={scaleIn}>
                <Card className="text-center border-none shadow-none bg-transparent">
                  <CardContent className="pt-6">
                    <stat.icon className="h-8 w-8 mx-auto mb-2 text-primary" />
                    <div className="text-3xl font-bold mb-1">{stat.value}</div>
                    <div className="text-sm text-muted-foreground">{stat.label}</div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeInUp}
            className="text-center max-w-2xl mx-auto mb-16"
          >
            <h2 className="text-white text-4xl font-bold mb-4">Everything You Need</h2>
            <p className="text-xl text-muted-foreground">
              Production-ready features to build enterprise-grade voice AI
            </p>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="grid md:grid-cols-3 gap-8"
          >
            {[
              {
                icon: Bot,
                title: 'Visual Flow Builder',
                description: 'Design conversation flows with our no-code drag-and-drop interface. No programming required.',
                color: 'text-blue-600 dark:text-blue-400',
              },
              {
                icon: Zap,
                title: 'Lightning Fast',
                description: 'Sub-2 second response times with optimized Gemini 2.5 integration and RAG pipeline.',
                color: 'text-amber-600 dark:text-amber-400',
              },
              {
                icon: Shield,
                title: 'Enterprise Security',
                description: 'Bank-level encryption, SOC 2 compliant, with complete audit trails and GDPR compliance.',
                color: 'text-emerald-600 dark:text-emerald-400',
              },
              {
                icon: MessageSquare,
                title: 'Advanced RAG',
                description: 'Hybrid search combining keyword and semantic search with automatic context retrieval.',
                color: 'text-violet-600 dark:text-violet-400',
              },
              {
                icon: BarChart3,
                title: 'Real-time Analytics',
                description: 'Monitor conversations, costs, sentiment, and performance in real-time dashboards.',
                color: 'text-rose-600 dark:text-rose-400',
              },
              {
                icon: DollarSign,
                title: 'Usage-Based Billing',
                description: 'Pay only for what you use. Transparent per-minute pricing with no hidden fees.',
                color: 'text-orange-600 dark:text-orange-400',
              },
            ].map((feature, i) => (
              <motion.div key={i} variants={scaleIn}>
                <Card className="h-full hover:shadow-lg transition-shadow">
                  <CardHeader>
                    <feature.icon className={`h-12 w-12 mb-4 ${feature.color}`} />
                    <CardTitle>{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-base">
                      {feature.description}
                    </CardDescription>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20 bg-muted/50">
        <div className="container mx-auto px-4">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeInUp}
            className="text-center max-w-2xl mx-auto mb-16"
          >
            <h2 className="text-white text-4xl font-bold mb-4">How It Works</h2>
            <p className="text-xl text-muted-foreground">
              Get your AI voice agent up and running in three simple steps
            </p>
          </motion.div>

          <div className="max-w-5xl mx-auto space-y-12">
            {[
              {
                step: '01',
                title: 'Design Your Flow',
                description: 'Use our visual builder to create conversation flows. Add prompts, tools, conditions, and handoffs with simple drag-and-drop.',
                image: '🎨',
                direction: 'left' as const,
              },
              {
                step: '02',
                title: 'Train with Knowledge',
                description: 'Upload documents, FAQs, and knowledge base. Our hybrid search automatically finds relevant context for every query.',
                image: '📚',
                direction: 'right' as const,
              },
              {
                step: '03',
                title: 'Deploy & Monitor',
                description: 'Connect your phone number and go live. Monitor conversations, costs, and sentiment in real-time dashboards.',
                image: '🚀',
                direction: 'left' as const,
              },
            ].map((item, i) => (
              <motion.div
                key={i}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                variants={item.direction === 'left' ? slideInLeft : slideInRight}
              >
                <Card className={`overflow-hidden ${item.direction === 'right' ? 'ml-auto' : ''} max-w-2xl`}>
                  <CardContent className="p-8">
                    <div className="flex items-start gap-6">
                      <div className="text-6xl">{item.image}</div>
                      <div className="flex-1">
                        <div className="text-sm font-mono text-primary mb-2">
                          Step {item.step}
                        </div>
                        <h3 className="text-2xl font-bold mb-2">{item.title}</h3>
                        <p className="text-muted-foreground">{item.description}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeInUp}
          >
            <Card className="border-2 border-primary bg-gradient-to-br from-primary/5 to-background">
              <CardContent className="p-12 text-center">
                <h2 className="text-4xl font-bold mb-4">
                  Ready to Transform Your Customer Experience?
                </h2>
                <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
                  Join thousands of companies using Velox AI to handle millions of conversations
                </p>
                <div className="flex gap-4 justify-center">
                  <Button size="lg" className="text-lg" asChild>
                    <Link to="/agents/demo/flow">
                      Start Free Trial
                      <ArrowRight className="ml-2 h-5 w-5" />
                    </Link>
                  </Button>
                  <Button size="lg" variant="outline" className="text-lg">
                    Schedule Demo
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t py-12">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Bot className="h-6 w-6 text-primary" />
                <span className="text-white **:font-bold">Velox AI</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Enterprise AI voice agent platform
              </p>
            </div>
            <div>
              <h4 className=" text-white font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><Link to="/features">Features</Link></li>
                <li><Link to="/billing">Pricing</Link></li>
                <li><Link to="/docs">Documentation</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="text-white font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><Link to="/about">About</Link></li>
                <li><Link to="/blog">Blog</Link></li>
                <li><Link to="/careers">Careers</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="text-white font-semibold mb-4">Legal</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><Link to="/privacy">Privacy</Link></li>
                <li><Link to="/terms">Terms</Link></li>
                <li><Link to="/security">Security</Link></li>
              </ul>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t text-center text-sm text-muted-foreground">
            © 2026 Velox AI. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  )
}