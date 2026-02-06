import React from 'react';

interface ConfidenceBadgeProps {
  score: number;
  category?: string;
}

/**
 * ConfidenceBadge - Visual indicator for prediction confidence levels
 * 
 * Provides color-coded badges that help clinicians quickly assess
 * the reliability of AI predictions at a glance.
 * 
 * Categories:
 * - High confidence (>0.9): Green - Can be used with confidence
 * - Medium confidence (0.7-0.9): Yellow - Consider as indicative
 * - Low confidence (<0.7): Red - Do not use for clinical decisions
 */
const ConfidenceBadge: React.FC<ConfidenceBadgeProps> = ({ score, category }) => {
  let color: string;
  let icon: string;
  let label: string;

  if (score >= 0.9) {
    color = 'bg-green-100 text-green-800 border-green-300';
    icon = '✅';
    label = 'Haute confiance';
  } else if (score >= 0.7) {
    color = 'bg-yellow-100 text-yellow-800 border-yellow-300';
    icon = '⚠️';
    label = 'Confiance moyenne';
  } else {
    color = 'bg-red-100 text-red-800 border-red-300';
    icon = '❌';
    label = 'Faible confiance';
  }
  
  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full border-2 text-sm font-medium ${color}`}>
      <span className="text-base" role="img" aria-label="Confidence level indicator">
        {icon}
      </span>
      <span>{label}</span>
      <span className="text-xs opacity-80 font-semibold">
        {(score * 100).toFixed(0)}%
      </span>
      {category && (
        <span className="text-xs opacity-70 ml-1">
          ({category})
        </span>
      )}
    </div>
  );
};

export default ConfidenceBadge;