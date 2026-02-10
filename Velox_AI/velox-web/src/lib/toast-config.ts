// src/lib/toast-config.ts

import { toast as sonnerToast } from 'sonner'

export const toast = {
  success: (message: string) => {
    sonnerToast.success(message, {
      duration: 3000,
      position: 'top-right',
      className: 'bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800',
    })
  },
  error: (message: string) => {
    sonnerToast.error(message, {
      duration: 4000,
      position: 'top-right',
      className: 'bg-red-50 dark:bg-red-950 border-red-200 dark:border-red-800',
    })
  },
  info: (message: string) => {
    sonnerToast.info(message, {
      duration: 3000,
      position: 'top-right',
    })
  },
}