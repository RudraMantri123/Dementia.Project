/**
 * Application constants
 */

export const AGENT_ICONS = {
  knowledge: 'BookOpen',
  empathy: 'Heart',
  cognitive: 'Brain',
  system: 'Activity',
};

export const AGENT_COLORS = {
  knowledge: 'bg-blue-100 text-blue-800',
  empathy: 'bg-pink-100 text-pink-800',
  cognitive: 'bg-purple-100 text-purple-800',
  system: 'bg-gray-100 text-gray-800',
};

export const SUGGESTED_TOPICS = [
  {
    id: 'knowledge',
    title: 'Learn About Dementia',
    description: 'Understand early signs, symptoms, and care strategies',
    prompt: 'What are the early signs and symptoms of dementia?',
    gradient: 'from-blue-50 to-blue-100 hover:from-blue-100 hover:to-blue-200',
    border: 'border-blue-200',
    iconBg: 'from-blue-500 to-blue-600',
    textColor: 'text-blue-900',
    descColor: 'text-blue-700',
    iconHover: 'text-blue-400 group-hover:text-blue-600',
    icon: 'BookOpen',
  },
  {
    id: 'empathy',
    title: 'Caregiver Support',
    description: 'Get emotional support and practical guidance',
    prompt: 'I need support with caregiving challenges',
    gradient: 'from-pink-50 to-pink-100 hover:from-pink-100 hover:to-pink-200',
    border: 'border-pink-200',
    iconBg: 'from-pink-500 to-pink-600',
    textColor: 'text-pink-900',
    descColor: 'text-pink-700',
    iconHover: 'text-pink-400 group-hover:text-pink-600',
    icon: 'Heart',
  },
  {
    id: 'cognitive',
    title: 'Cognitive Exercises',
    description: 'Practice memory and brain training activities',
    prompt: 'Please provide a cognitive exercise',
    gradient: 'from-purple-50 to-purple-100 hover:from-purple-100 hover:to-purple-200',
    border: 'border-purple-200',
    iconBg: 'from-purple-500 to-purple-600',
    textColor: 'text-purple-900',
    descColor: 'text-purple-700',
    iconHover: 'text-purple-400 group-hover:text-purple-600',
    icon: 'Brain',
  },
];

export const DEFAULT_SESSION_ID = 'default';
