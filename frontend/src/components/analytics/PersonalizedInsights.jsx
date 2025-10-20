import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  Clock, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle, 
  Target,
  Users,
  Heart,
  Calendar,
  Star,
  Activity,
  Zap,
  Stethoscope,
  Shield,
  Eye
} from 'lucide-react';

const PersonalizedInsights = ({ sessionId = 'default' }) => {
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchPersonalizedInsights();
  }, [sessionId]);

  const fetchPersonalizedInsights = async () => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:8000/personalized-insights/${sessionId}`);
      const data = await response.json();
      
      if (data.status === 'success') {
        setInsights(data.personalized_insights);
      } else {
        setError('Failed to fetch personalized insights');
      }
    } catch (err) {
      setError('Error fetching personalized insights');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'critical': return 'text-red-600 dark:text-red-400';
      case 'high': return 'text-orange-600 dark:text-orange-400';
      case 'medium': return 'text-yellow-600 dark:text-yellow-400';
      case 'low': return 'text-green-600 dark:text-green-400';
      default: return 'text-gray-600 dark:text-gray-400';
    }
  };

  const getPriorityIcon = (priority) => {
    switch (priority) {
      case 'critical': return <AlertTriangle className="w-4 h-4" />;
      case 'high': return <Zap className="w-4 h-4" />;
      case 'medium': return <Target className="w-4 h-4" />;
      case 'low': return <CheckCircle className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  const getUrgencyColor = (urgency) => {
    switch (urgency) {
      case 'immediate': return 'bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-200';
      case 'high': return 'bg-orange-100 dark:bg-orange-900/20 text-orange-800 dark:text-orange-200';
      case 'medium': return 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-200';
      case 'low': return 'bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-200';
      default: return 'bg-gray-100 dark:bg-gray-900/20 text-gray-800 dark:text-gray-200';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600"></div>
        <span className="ml-2 text-gray-600 dark:text-gray-400">Analyzing your patterns...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 text-center">
        <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
        <p className="text-red-600 dark:text-red-400">{error}</p>
        <button 
          onClick={fetchPersonalizedInsights}
          className="mt-4 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
        >
          Try Again
        </button>
      </div>
    );
  }

  if (!insights) {
    return (
      <div className="p-6 text-center">
        <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-600 dark:text-gray-400">No personalized insights available yet.</p>
        <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">
          Continue the conversation to generate personalized insights.
        </p>
      </div>
    );
  }

  const personalizedAnalysis = insights.personalized_analysis || {};
  const patterns = personalizedAnalysis.patterns || {};
  const keyInsights = insights.key_insights || [];
  const actionItems = insights.action_items || [];
  const supportPriorities = insights.support_priorities || [];
  const personalizedRecommendations = insights.personalized_recommendations || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 p-6 rounded-lg">
        <div className="flex items-center gap-3 mb-2">
          <Brain className="w-6 h-6 text-purple-600 dark:text-purple-400" />
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">
            Your Personalized Insights
          </h2>
        </div>
        <p className="text-gray-600 dark:text-gray-400">
          AI-powered analysis of your unique patterns, preferences, and support needs
        </p>
        <div className="mt-3 flex items-center gap-4 text-sm">
          <div className="flex items-center gap-1">
            <Star className="w-4 h-4 text-yellow-500" />
            <span className="text-gray-600 dark:text-gray-400">
              Confidence: {Math.round((insights.insight_confidence || 0) * 100)}%
            </span>
          </div>
          <div className="flex items-center gap-1">
            <Calendar className="w-4 h-4 text-blue-500" />
            <span className="text-gray-600 dark:text-gray-400">
              {new Date(insights.analysis_timestamp).toLocaleDateString()}
            </span>
          </div>
        </div>
      </div>

      {/* Key Insights */}
      {keyInsights.length > 0 && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            Key Insights About You
          </h3>
          <div className="space-y-3">
            {keyInsights.map((insight, index) => (
              <div key={index} className="flex items-start gap-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <CheckCircle className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
                <p className="text-gray-700 dark:text-gray-300">{insight}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Personalized Recommendations */}
      {personalizedRecommendations.length > 0 && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <Target className="w-5 h-5 text-green-600 dark:text-green-400" />
            Personalized Recommendations
          </h3>
          <div className="space-y-4">
            {personalizedRecommendations.map((rec, index) => (
              <div key={index} className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0">
                    {rec.urgency === 'immediate' ? (
                      <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400" />
                    ) : (
                      <Heart className="w-5 h-5 text-green-600 dark:text-green-400" />
                    )}
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-900 dark:text-white mb-1">
                      {rec.title}
                    </h4>
                    <p className="text-gray-700 dark:text-gray-300 mb-2">
                      {rec.description}
                    </p>
                    <div className="flex items-center gap-4 text-sm">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getUrgencyColor(rec.urgency || 'low')}`}>
                        {rec.urgency || 'low'} priority
                      </span>
                      <span className="text-gray-500 dark:text-gray-400">
                        Confidence: {Math.round((rec.confidence || 0) * 100)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Action Items */}
      {actionItems.length > 0 && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <Calendar className="w-5 h-5 text-orange-600 dark:text-orange-400" />
            Your Action Plan
          </h3>
          <div className="space-y-3">
            {actionItems.map((item, index) => (
              <div key={index} className="flex items-start gap-3 p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200 dark:border-orange-800">
                <div className="flex-shrink-0">
                  {getPriorityIcon(item.priority)}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-semibold text-gray-900 dark:text-white">
                      {item.action}
                    </h4>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getUrgencyColor(item.priority)}`}>
                      {item.priority}
                    </span>
                  </div>
                  <p className="text-gray-700 dark:text-gray-300 mb-2">
                    {item.description}
                  </p>
                  <div className="flex items-center gap-1 text-sm text-gray-500 dark:text-gray-400">
                    <Clock className="w-4 h-4" />
                    <span>Timeline: {item.timeline}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Support Priorities */}
      {supportPriorities.length > 0 && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <Users className="w-5 h-5 text-purple-600 dark:text-purple-400" />
            Support Priorities
          </h3>
          <div className="space-y-3">
            {supportPriorities.map((priority, index) => (
              <div key={index} className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 bg-purple-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                      {priority.priority_level}
                    </div>
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-900 dark:text-white mb-1">
                      {priority.description}
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2 capitalize">
                      Category: {priority.category.replace('_', ' ')}
                    </p>
                    <div className="flex items-center gap-2 mb-2">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getUrgencyColor(priority.urgency)}`}>
                        {priority.urgency} urgency
                      </span>
                    </div>
                    <div className="mt-2">
                      <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Resources:</p>
                      <div className="flex flex-wrap gap-1">
                        {priority.resources.map((resource, idx) => (
                          <span key={idx} className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded text-xs">
                            {resource}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Clinical Analysis */}
      {patterns.dementia_analysis && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <Stethoscope className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            Clinical Pattern Analysis
          </h3>
          
          {(() => {
            const dementiaAnalysis = patterns.dementia_analysis;
            const assessment = dementiaAnalysis.assessment;
            const clinicalGuidance = dementiaAnalysis.clinical_guidance;
            
            return (
              <div className="space-y-4">
                {/* Assessment Summary */}
                <div className={`p-4 rounded-lg border ${
                  assessment === 'concerning_patterns' ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800' :
                  assessment === 'mixed_patterns' ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800' :
                  assessment === 'normal_aging' ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800' :
                  'bg-gray-50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-800'
                }`}>
                  <div className="flex items-center gap-2 mb-2">
                    {assessment === 'concerning_patterns' ? (
                      <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400" />
                    ) : assessment === 'mixed_patterns' ? (
                      <Eye className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
                    ) : assessment === 'normal_aging' ? (
                      <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
                    ) : (
                      <Shield className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                    )}
                    <h4 className="font-semibold text-gray-900 dark:text-white capitalize">
                      {assessment.replace('_', ' ')} Assessment
                    </h4>
                  </div>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                    {dementiaAnalysis.recommendation}
                  </p>
                  <div className="flex items-center gap-4 text-xs">
                    <span className={`px-2 py-1 rounded-full font-medium ${
                      clinicalGuidance.urgency === 'high' ? 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-200' :
                      clinicalGuidance.urgency === 'moderate' ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-200' :
                      'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200'
                    }`}>
                      {clinicalGuidance.urgency} urgency
                    </span>
                    <span className="text-gray-500 dark:text-gray-400">
                      Timeline: {clinicalGuidance.timeline}
                    </span>
                  </div>
                </div>

                {/* Scores */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <h5 className="font-medium text-gray-900 dark:text-white text-sm mb-1">Dementia Indicators</h5>
                    <p className="text-lg font-bold text-blue-600 dark:text-blue-400">
                      {dementiaAnalysis.total_dementia_score}
                    </p>
                    <p className="text-xs text-gray-600 dark:text-gray-400">out of 5 categories</p>
                  </div>
                  <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <h5 className="font-medium text-gray-900 dark:text-white text-sm mb-1">Normal Aging Indicators</h5>
                    <p className="text-lg font-bold text-green-600 dark:text-green-400">
                      {dementiaAnalysis.total_normal_aging_score}
                    </p>
                    <p className="text-xs text-gray-600 dark:text-gray-400">out of 5 categories</p>
                  </div>
                </div>

                {/* Clinical Resources */}
                {clinicalGuidance.resources && clinicalGuidance.resources.length > 0 && (
                  <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <h5 className="font-medium text-gray-900 dark:text-white mb-2">Recommended Actions</h5>
                    <ul className="space-y-1">
                      {clinicalGuidance.resources.map((resource, index) => (
                        <li key={index} className="text-sm text-gray-700 dark:text-gray-300 flex items-start gap-2">
                          <CheckCircle className="w-3 h-3 text-blue-600 dark:text-blue-400 mt-1 flex-shrink-0" />
                          {resource}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Watch For / Red Flags */}
                {(clinicalGuidance.watch_for || clinicalGuidance.red_flags) && (
                  <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                    <h5 className="font-medium text-gray-900 dark:text-white mb-2">
                      {clinicalGuidance.red_flags ? 'Red Flags to Monitor' : 'Watch For'}
                    </h5>
                    <ul className="space-y-1">
                      {(clinicalGuidance.watch_for || clinicalGuidance.red_flags).map((item, index) => (
                        <li key={index} className="text-sm text-gray-700 dark:text-gray-300 flex items-start gap-2">
                          <AlertTriangle className="w-3 h-3 text-orange-600 dark:text-orange-400 mt-1 flex-shrink-0" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Reassurance for Normal Aging */}
                {clinicalGuidance.reassurance && (
                  <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <h5 className="font-medium text-gray-900 dark:text-white mb-2">Reassurance</h5>
                    <ul className="space-y-1">
                      {clinicalGuidance.reassurance.map((item, index) => (
                        <li key={index} className="text-sm text-gray-700 dark:text-gray-300 flex items-start gap-2">
                          <CheckCircle className="w-3 h-3 text-green-600 dark:text-green-400 mt-1 flex-shrink-0" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            );
          })()}
        </div>
      )}

      {/* Pattern Analysis */}
      {patterns.temporal_patterns && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
            Your Patterns
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {patterns.temporal_patterns && (
              <div className="p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg">
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Engagement Pattern</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 capitalize">
                  {patterns.temporal_patterns.frequency_pattern?.replace('_', ' ')}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400 capitalize">
                  {patterns.temporal_patterns.time_preference?.replace('_', ' ')}
                </p>
              </div>
            )}
            {patterns.emotional_patterns && (
              <div className="p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg">
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Emotional Pattern</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 capitalize">
                  Dominant: {patterns.emotional_patterns.dominant_emotion}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Volatility: {Math.round((patterns.emotional_patterns.emotional_volatility || 0) * 100)}%
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Refresh Button */}
      <div className="text-center">
        <button 
          onClick={fetchPersonalizedInsights}
          className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center gap-2 mx-auto"
        >
          <Brain className="w-4 h-4" />
          Refresh Insights
        </button>
      </div>
    </div>
  );
};

export default PersonalizedInsights;
