import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { useUser } from '@clerk/clerk-react';
import type { Policy } from '../types';

export function usePolicies() {
  const { user } = useUser();
  const userEmail = user?.primaryEmailAddress?.emailAddress || "unknown";

  return useQuery({
    // Unique key for caching (cache is specific to this email)
    queryKey: ['policies', userEmail], 
    
    // The fetch function
    queryFn: async (): Promise<Policy[]> => {
      // We pass the email as a query param so the backend knows who is asking
      const res = await axios.get(`/api/policies?user_email=${userEmail}`);
      return res.data;
    },
    
    // Only run this query if we actually have a user loaded
    enabled: !!userEmail,
  });
}