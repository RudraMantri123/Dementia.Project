import React from 'react';
import { Brain, Heart, BookOpen } from 'lucide-react';
import { SUGGESTED_TOPICS } from '../../utils/constants';

const iconMap = {
  Brain,
  Heart,
  BookOpen,
};

const WelcomeScreen = ({ onSelectTopic, hasMessages }) => {
  if (hasMessages) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-6">
        <div className="mb-8">
          <div className="relative mb-6">
            <Brain className="w-16 h-16 text-blue-600 mx-auto animate-pulse" />
          </div>
          <h3 className="text-2xl font-bold text-gray-900 mb-3 text-xl-large">
            Welcome to Dementia Support
          </h3>
          <p className="text-gray-600 mb-8 text-large">
            Ask a question or explore suggested topics below
          </p>
        </div>

        <div className="space-y-4 text-left max-w-2xl w-full">
          {SUGGESTED_TOPICS.map((topic) => {
            const IconComponent = iconMap[topic.icon];

            return (
              <div
                key={topic.id}
                onClick={() => onSelectTopic(topic.prompt)}
                className={`suggested-topic bg-gradient-to-r ${topic.gradient} ${topic.border} group cursor-pointer`}
              >
                <div className="flex items-start gap-4">
                  <div className={`w-12 h-12 bg-gradient-to-br ${topic.iconBg} rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300 flex-shrink-0`}>
                    <IconComponent className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1">
                    <h4 className={`text-lg font-bold ${topic.textColor} mb-2`}>
                      {topic.title}
                    </h4>
                    <p className={`${topic.descColor} text-large`}>
                      {topic.description}
                    </p>
                  </div>
                  <div className={`${topic.iconHover} transition-colors duration-300`}>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center h-full text-center p-8 bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="mb-8">
        <div className="relative">
          <Brain className="w-20 h-20 text-blue-600 mb-6 mx-auto animate-pulse" />
        </div>
        <h1 className="text-4xl font-bold text-gray-900 mb-4 text-xl-large">
          Welcome to Dementia Support
        </h1>
        <p className="text-gray-600 mb-8 max-w-lg text-large">
          Your compassionate AI companion for dementia care and support
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl w-full">
        <div className="card-interactive group">
          <div className="flex flex-col items-center text-center">
            <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
              <BookOpen className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-3">Knowledge Agent</h3>
            <p className="text-gray-600 text-large">
              Ask factual questions about dementia, symptoms, and caregiving
            </p>
          </div>
        </div>

        <div className="card-interactive group">
          <div className="flex flex-col items-center text-center">
            <div className="w-16 h-16 bg-gradient-to-br from-pink-500 to-pink-600 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
              <Heart className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-3">Empathy Agent</h3>
            <p className="text-gray-600 text-large">
              Share your feelings and receive emotional support
            </p>
          </div>
        </div>

        <div className="card-interactive group">
          <div className="flex flex-col items-center text-center">
            <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-purple-600 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-3">Cognitive Agent</h3>
            <p className="text-gray-600 text-large">
              Practice memory exercises and brain training activities
            </p>
          </div>
        </div>
      </div>

      <div className="mt-12 p-6 bg-gradient-to-r from-blue-100 to-purple-100 rounded-2xl border-2 border-blue-200">
        <p className="text-gray-700 text-large font-medium">
          [Tip] <strong>Getting Started:</strong> Configure your API key in the sidebar to begin your journey
        </p>
      </div>
    </div>
  );
};

export default WelcomeScreen;
