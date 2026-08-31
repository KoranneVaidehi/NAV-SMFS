export interface VerificationResult {
  score: number;
  label: 'LIKELY AUTHENTIC' | 'LIKELY MANIPULATED';
  facialConsistency: 'Strong' | 'Medium' | 'Low' | 'High';
  visualConsistency?: 'Strong' | 'Medium' | 'Low';
  textureAnomaly?: 'High' | 'Medium' | 'Low';
  localRegionAnomaly?: 'High' | 'Medium' | 'Low';
  forensicAnomalyLevel?: 'High' | 'Medium' | 'Low';
  explanation: string;
}

export const analyzeImage = async (_file: File): Promise<VerificationResult> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Mock logic: randomly decide authentic or manipulated for demonstration
      const isAuthentic = Math.random() > 0.5;
      
      if (isAuthentic) {
        resolve({
          score: 92.8,
          label: 'LIKELY AUTHENTIC',
          facialConsistency: 'Strong',
          visualConsistency: 'Strong',
          forensicAnomalyLevel: 'Low',
          explanation: "The analyzed image shows stronger consistency with an authentic image than with the manipulated-image patterns evaluated by the current system."
        });
      } else {
        resolve({
          score: 87.4,
          label: 'LIKELY MANIPULATED',
          facialConsistency: 'High', // High inconsistency
          textureAnomaly: 'High',
          localRegionAnomaly: 'Medium',
          explanation: "The analyzed image exhibits spatial anomalies and texture inconsistencies indicative of manipulation or synthetic generation."
        });
      }
    }, 4000); // 4 seconds total processing time to allow UI stages to show
  });
};
