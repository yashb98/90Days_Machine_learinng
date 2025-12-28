import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, CheckCircle, FileText, Database, Cpu, Shield } from 'lucide-react';

interface UploadProgressProps {
  isUploading: boolean;
  filename: string;
}

const STEPS = [
  { label: "Securing Connection...", icon: Shield, duration: 800 },
  { label: "Uploading Payload...", icon: FileText, duration: 1500 },
  { label: "Chunking Document...", icon: Cpu, duration: 2000 },
  { label: "Vectorizing Content...", icon: Database, duration: 2500 },
  { label: "Indexing Memories...", icon: CheckCircle, duration: 1500 },
];

export function UploadProgress({ isUploading, filename }: UploadProgressProps) {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    if (isUploading) {
      setCurrentStep(0);
      let stepIndex = 0;
      
      const interval = setInterval(() => {
        stepIndex++;
        if (stepIndex < STEPS.length) {
          setCurrentStep(stepIndex);
        }
      }, 1500); // Change text every 1.5s to simulate progress

      return () => clearInterval(interval);
    }
  }, [isUploading]);

  if (!isUploading) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        className="absolute bottom-full left-0 w-full mb-4 px-6"
      >
        <div className="bg-terminal-dark border border-neon-blue/30 rounded-lg p-4 shadow-[0_0_20px_rgba(59,130,246,0.2)] backdrop-blur-md">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-neon-blue/10 rounded-full">
                {currentStep === STEPS.length - 1 ? (
                  <CheckCircle className="w-5 h-5 text-neon-green" />
                ) : (
                  <Loader2 className="w-5 h-5 text-neon-blue animate-spin" />
                )}
              </div>
              <div>
                <div className="text-sm font-bold text-white tracking-wider">
                   {STEPS[currentStep].label}
                </div>
                <div className="text-[10px] text-gray-400 font-mono">
                   TARGET: {filename}
                </div>
              </div>
            </div>
            <div className="text-neon-blue font-mono text-xs">
              {Math.min((currentStep + 1) * 20, 99)}%
            </div>
          </div>

          {/* Progress Bar Track */}
          <div className="h-1 w-full bg-gray-800 rounded-full overflow-hidden">
            <motion.div 
              className="h-full bg-neon-blue shadow-[0_0_10px_#3b82f6]"
              initial={{ width: "0%" }}
              animate={{ width: `${((currentStep + 1) / STEPS.length) * 100}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
