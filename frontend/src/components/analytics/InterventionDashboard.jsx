import React, { useState, useEffect } from 'react';
import { AlertTriangle, Brain, Target, Clock, TrendingUp, Shield, Zap } from 'lucide-react';

const InterventionDashboard = ({ sessionId = 'default' }) => {
  const [interventionPlan, setInterventionPlan] = useState(null);
  const [riskAssessment, setRiskAssessment] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchInterventionData();
    // Refresh every 30 seconds
    const interval = setInterval(fetchInterventionData, 30000);
    return () => clearInterval(interval);
  }, [sessionId]);

  const fetchInterventionData = async () => {
    try {
      setLoading(true);
      
      const [planResponse, riskResponse] = await Promise.all([
        fetch(`http://localhost:8000/ml/intervention-plan/${sessionId}`),
        fetch(`http://localhost:8000/ml/risk-assessment/${sessionId}`)
      ]);

      if (!planResponse.ok || !riskResponse.ok) {
        throw new Error('Failed to fetch intervention data');
      }

      const planData = await planResponse.json();
      const riskData = await riskResponse.json();

      setInterventionPlan(planData.intervention_plan);
      setRiskAssessment(riskData.risk_assessment);
      setError(null);
    } catch (err) {
      console.error('Error fetching intervention data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (level) => {
    const colors = {
      critical: 'text-red-600 bg-red-100 border-red-300 dark:text-red-400 dark:bg-red-900/30 dark:border-red-700',
      high: 'text-orange-600 bg-orange-100 border-orange-300 dark:text-orange-400 dark:bg-orange-900/30 dark:border-orange-700',
      moderate: 'text-yellow-600 bg-yellow-100 border-yellow-300 dark:text-yellow-400 dark:bg-yellow-900/30 dark:border-yellow-700',
      low: 'text-green-600 bg-green-100 border-green-300 dark:text-green-400 dark:bg-green-900/30 dark:border-green-700',
      unknown: 'text-gray-600 bg-gray-100 border-gray-300 dark:text-gray-400 dark:bg-gray-800 dark:border-gray-600'
    };
    return colors[level] || colors.unknown;
  };

  const getPriorityColor = (priority) => {
    const colors = {
      immediate: 'bg-red-600 text-white',
      urgent: 'bg-orange-600 text-white',
      elevated: 'bg-yellow-600 text-white',
      routine: 'bg-green-600 text-white'
    };
    return colors[priority] || 'bg-gray-600 text-white';
  };

  if (loading && !interventionPlan) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-300 dark:border-red-700 rounded-lg p-6">
        <p className="text-red-800 dark:text-red-400">Error: {error}</p>
      </div>
    );
  }

  if (!interventionPlan || !riskAssessment) {
    return (
      <div className="bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg p-6">
        <p className="text-gray-600 dark:text-gray-400">No intervention data available. Start a conversation to see insights.</p>
      </div>
    );
  }

  const { intervention_plan, sentiment_analysis, personalized_recommendations, optimal_timing } = interventionPlan;
  const { burnout_risk, crisis_risk } = riskAssessment;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
          <Brain className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          ML-Powered Intervention Plan
        </h2>
        <button
          onClick={fetchInterventionData}
          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
        >
          Refresh
        </button>
      </div>

      {/* Priority Alert */}
      {intervention_plan && (
        <div className={`${getPriorityColor(intervention_plan.priority)} rounded-lg p-4 shadow-md`}>
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-6 h-6" />
            <div>
              <h3 className="font-bold text-lg">Priority: {intervention_plan.priority.toUpperCase()}</h3>
              <p className="text-sm">{intervention_plan.action}</p>
            </div>
          </div>
        </div>
      )}

      {/* Risk Assessment Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Burnout Risk */}
        <div className={`border-2 rounded-lg p-6 ${getRiskColor(burnout_risk.risk_level)}`}>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold flex items-center gap-2">
              <Shield className="w-5 h-5" />
              Burnout Risk
            </h3>
            <span className="text-2xl font-bold">{(burnout_risk.probability * 100).toFixed(0)}%</span>
          </div>
          <div className="space-y-2">
            <p className="font-semibold">Risk Level: {burnout_risk.risk_level.toUpperCase()}</p>
            {burnout_risk.factors && burnout_risk.factors.length > 0 && (
              <div>
                <p className="text-sm font-semibold mb-1">Key Factors:</p>
                <ul className="text-sm space-y-1">
                  {burnout_risk.factors.slice(0, 3).map((factor, idx) => (
                    <li key={idx} className="flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-current"></span>
                      {factor.replace(/_/g, ' ')}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>

        {/* Crisis Risk */}
        <div className={`border-2 rounded-lg p-6 ${crisis_risk.is_crisis ? getRiskColor('critical') : getRiskColor('low')}`}>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              Crisis Risk
            </h3>
            <span className="text-2xl font-bold">{(crisis_risk.probability * 100).toFixed(0)}%</span>
          </div>
          <div className="space-y-2">
            <p className="font-semibold">Status: {crisis_risk.is_crisis ? 'ALERT' : 'Normal'}</p>
            <p className="text-sm">Urgency: {crisis_risk.urgency.toUpperCase()}</p>
          </div>
        </div>
      </div>

      {/* Recommended Intervention */}
      {intervention_plan.recommended_intervention && (
        <div className="bg-gradient-to-br from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 border border-purple-300 dark:border-purple-700 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-4">
            <Target className="w-6 h-6 text-purple-600 dark:text-purple-400" />
            <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100">Recommended Intervention</h3>
          </div>
          
          <div className="space-y-4">
            <div>
              <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                {intervention_plan.recommended_intervention.details.name}
              </p>
              <p className="text-gray-700 dark:text-gray-300 mt-1">
                {intervention_plan.recommended_intervention.details.description}
              </p>
            </div>

            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
              <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                Confidence: {(intervention_plan.recommended_intervention.confidence * 100).toFixed(0)}%
              </span>
            </div>

            {intervention_plan.recommended_intervention.details.techniques && (
              <div>
                <p className="font-semibold text-gray-900 dark:text-gray-100 mb-2">Recommended Techniques:</p>
                <div className="flex flex-wrap gap-2">
                  {intervention_plan.recommended_intervention.details.techniques.map((technique, idx) => (
                    <span
                      key={idx}
                      className="px-3 py-1 bg-purple-200 dark:bg-purple-800 text-purple-900 dark:text-purple-200 rounded-full text-sm"
                    >
                      {technique.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Sentiment Analysis */}
      {sentiment_analysis && (
        <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg p-6">
          <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            Current Emotional State
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Overall Sentiment</p>
              <p className="text-lg font-bold text-gray-900 dark:text-gray-100 capitalize">
                {sentiment_analysis.overall_sentiment}
              </p>
            </div>
            <div className="text-center">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Intensity</p>
              <p className="text-lg font-bold text-gray-900 dark:text-gray-100">
                {sentiment_analysis.emotional_intensity?.toFixed(1)}/10
              </p>
            </div>
            <div className="text-center">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Stability</p>
              <p className="text-lg font-bold text-gray-900 dark:text-gray-100">
                {(sentiment_analysis.sentiment_stability * 100)?.toFixed(0)}%
              </p>
            </div>
            <div className="text-center">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Volatility</p>
              <p className="text-lg font-bold text-gray-900 dark:text-gray-100">
                {sentiment_analysis.emotional_volatility?.toFixed(2)}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Personalized Recommendations */}
      {personalized_recommendations && (
        <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg p-6">
          <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-4">Personalized Recommendations</h3>
          
          <div className="space-y-3">
            {personalized_recommendations.techniques && personalized_recommendations.techniques.length > 0 && (
              <div>
                <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">Preferred Techniques:</p>
                <div className="flex flex-wrap gap-2">
                  {personalized_recommendations.techniques.map((technique, idx) => (
                    <span
                      key={idx}
                      className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-900 dark:text-blue-300 rounded-full text-sm"
                    >
                      {technique}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {personalized_recommendations.focus_areas && personalized_recommendations.focus_areas.length > 0 && (
              <div>
                <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">Focus Areas:</p>
                <div className="flex flex-wrap gap-2">
                  {personalized_recommendations.focus_areas.map((area, idx) => (
                    <span
                      key={idx}
                      className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-900 dark:text-green-300 rounded-full text-sm"
                    >
                      {area.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {personalized_recommendations.communication_approach && (
              <p className="text-sm text-gray-700 dark:text-gray-300">
                <span className="font-semibold">Communication Style:</span> {personalized_recommendations.communication_approach}
              </p>
            )}

            {personalized_recommendations.primary_need && (
              <p className="text-sm text-gray-700 dark:text-gray-300">
                <span className="font-semibold">Primary Need:</span> {personalized_recommendations.primary_need.replace(/_/g, ' ')}
              </p>
            )}
          </div>
        </div>
      )}

      {/* Timing Optimization */}
      {optimal_timing && (
        <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg p-6">
          <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
            Optimal Timing
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Next Check-in</p>
              <p className="font-semibold text-gray-900 dark:text-gray-100">{optimal_timing.next_checkin}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Best Time</p>
              <p className="font-semibold text-gray-900 dark:text-gray-100 capitalize">{optimal_timing.optimal_time}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Frequency</p>
              <p className="font-semibold text-gray-900 dark:text-gray-100">{optimal_timing.frequency.replace(/_/g, ' ')}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default InterventionDashboard;

