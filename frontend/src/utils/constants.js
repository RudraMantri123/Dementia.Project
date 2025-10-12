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
  knowledge: 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200',
  empathy: 'bg-pink-100 dark:bg-pink-900/30 text-pink-800 dark:text-pink-200',
  cognitive: 'bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-200',
  system: 'bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200',
};

export const SUGGESTED_TOPICS = [
  {
    id: 'knowledge',
    title: 'Learn About Dementia',
    description: 'Understand early signs, symptoms, and care strategies',
    prompt: 'What are the early signs and symptoms of dementia?',
    gradient: 'from-blue-50 to-blue-100 hover:from-blue-100 hover:to-blue-200 dark:from-blue-900/30 dark:to-blue-800/40 dark:hover:from-blue-800/40 dark:hover:to-blue-700/50',
    border: 'border-blue-200 dark:border-blue-700',
    iconBg: 'from-blue-500 to-blue-600',
    textColor: 'text-blue-900 dark:text-blue-100',
    descColor: 'text-blue-700 dark:text-blue-300',
    iconHover: 'text-blue-400 group-hover:text-blue-600 dark:text-blue-300 dark:group-hover:text-blue-400',
    icon: 'BookOpen',
  },
  {
    id: 'empathy',
    title: 'Caregiver Support',
    description: 'Get emotional support and practical guidance',
    prompt: 'I need support with caregiving challenges',
    gradient: 'from-pink-50 to-pink-100 hover:from-pink-100 hover:to-pink-200 dark:from-pink-900/30 dark:to-pink-800/40 dark:hover:from-pink-800/40 dark:hover:to-pink-700/50',
    border: 'border-pink-200 dark:border-pink-700',
    iconBg: 'from-pink-500 to-pink-600',
    textColor: 'text-pink-900 dark:text-pink-100',
    descColor: 'text-pink-700 dark:text-pink-300',
    iconHover: 'text-pink-400 group-hover:text-pink-600 dark:text-pink-300 dark:group-hover:text-pink-400',
    icon: 'Heart',
  },
  {
    id: 'cognitive',
    title: 'Cognitive Exercises',
    description: 'Practice memory and brain training activities',
    prompt: 'Please provide a cognitive exercise',
    gradient: 'from-purple-50 to-purple-100 hover:from-purple-100 hover:to-purple-200 dark:from-purple-900/30 dark:to-purple-800/40 dark:hover:from-purple-800/40 dark:hover:to-purple-700/50',
    border: 'border-purple-200 dark:border-purple-700',
    iconBg: 'from-purple-500 to-purple-600',
    textColor: 'text-purple-900 dark:text-purple-100',
    descColor: 'text-purple-700 dark:text-purple-300',
    iconHover: 'text-purple-400 group-hover:text-purple-600 dark:text-purple-300 dark:group-hover:text-purple-400',
    icon: 'Brain',
  },
];

export const DEFAULT_SESSION_ID = 'default';
