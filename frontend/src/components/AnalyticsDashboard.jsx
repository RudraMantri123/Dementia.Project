import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LineChart, Line, PieChart, Pie, Cell as PieCell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { TrendingUp, MessageSquare, Brain, Heart, BookOpen, Activity, Loader2, BarChart3, PieChart as PieChartIcon, Zap, Shield, AlertTriangle, CheckCircle } from 'lucide-react';

const sentimentColors = {
  positive: '#10b981',
  neutral: '#6b7280',
  stressed: '#ef4444',
  sad: '#3b82f6',
  anxious: '#f59e0b',
  frustrated: '#f97316',
};

const agentIcons = {
  knowledge: <BookOpen className="w-4 h-4" />,
  empathy: <Heart className="w-4 h-4" />,
  cognitive: <Brain className="w-4 h-4" />,
  system: <Activity className="w-4 h-4" />,
};

const AnalyticsDashboard = ({ stats, analytics, isLoading }) => {
  const [chartData, setChartData] = useState([]);
  const [intensityData, setIntensityData] = useState([]);
  const [confidenceData, setConfidenceData] = useState([]);
  const [radarData, setRadarData] = useState([]);

  useEffect(() => {
    if (analytics?.sentiment_distribution) {
      // Enhanced sentiment distribution data
      const data = Object.entries(analytics.sentiment_distribution).map(([key, value]) => ({
        sentiment: key,
        count: value,
        color: sentimentColors[key] || '#6b7280',
        percentage: analytics.detailed_metrics?.sentiment_breakdown?.[key]?.percentage || 0,
        confidence: analytics.detailed_metrics?.sentiment_breakdown?.[key]?.avg_confidence || 0,
        intensity: analytics.detailed_metrics?.sentiment_breakdown?.[key]?.avg_intensity || 0,
      }));
      setChartData(data);

      // Emotional intensity data
      if (analytics.detailed_metrics?.sentiment_breakdown) {
        const intensityData = Object.entries(analytics.detailed_metrics.sentiment_breakdown).map(([emotion, data]) => ({
          emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
          intensity: data.avg_intensity,
          confidence: data.avg_confidence,
          count: data.count
        }));
        setIntensityData(intensityData);
      }

      // Confidence metrics data
      if (analytics.detailed_metrics?.confidence_metrics) {
        const confMetrics = analytics.detailed_metrics.confidence_metrics;
        setConfidenceData([
          { metric: 'Average', value: confMetrics.average * 100 },
          { metric: 'Min', value: confMetrics.min * 100 },
          { metric: 'Max', value: confMetrics.max * 100 },
          { metric: 'Stability', value: (1 - confMetrics.std) * 100 }
        ]);
      }

      // Radar chart data for comprehensive analysis
      if (analytics.detailed_metrics?.sentiment_breakdown) {
        const radarData = Object.entries(analytics.detailed_metrics.sentiment_breakdown).map(([emotion, data]) => ({
          emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
          intensity: data.avg_intensity * 10, // Scale to 0-100
          confidence: data.avg_confidence * 100,
          frequency: data.percentage
        }));
        setRadarData(radarData);
      }
    }
  }, [analytics]);

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 text-center">
        <Loader2 className="w-16 h-16 text-primary-600 mb-4 animate-spin" />
        <p className="text-xl font-semibold text-gray-900 mb-2">
          Analyzing Your Conversation...
        </p>
        <div className="space-y-1 text-sm text-gray-600">
          <p>Processing {stats?.total_messages || 0} messages with advanced ML sentiment analysis...</p>
          <p>Calculating emotional intensity and confidence metrics...</p>
          <p>Generating comprehensive insights and recommendations...</p>
        </div>
      </div>
    );
  }

  if (!stats || stats.total_messages === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 text-center">
        <TrendingUp className="w-12 h-12 text-gray-400 mb-4" />
        <p className="text-gray-600">
          Start a conversation to see detailed analytics
        </p>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2 flex items-center gap-2">
          <BarChart3 className="w-6 h-6" />
          Advanced Conversation Analytics
        </h2>
        <p className="text-sm text-gray-600">Comprehensive ML-powered emotional analysis and insights</p>
      </div>

      {/* Enhanced Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Messages</p>
              <p className="text-3xl font-bold text-gray-900 mt-1">
                {stats.total_messages}
              </p>
            </div>
            <MessageSquare className="w-8 h-8 text-primary-600" />
          </div>
        </div>

        {analytics && (
          <>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Emotional Intensity</p>
                  <p className="text-3xl font-bold text-gray-900 mt-1">
                    {analytics.emotional_intensity?.toFixed(1) || '0.0'}/10
                  </p>
                </div>
                <Zap className="w-8 h-8 text-yellow-600" />
              </div>
            </div>

            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Analysis Confidence</p>
                  <p className="text-3xl font-bold text-gray-900 mt-1">
                    {((analytics.sentiment_confidence || 0) * 100).toFixed(0)}%
                  </p>
                </div>
                <Shield className="w-8 h-8 text-green-600" />
              </div>
            </div>

            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Sentiment Stability</p>
                  <p className="text-3xl font-bold text-gray-900 mt-1">
                    {((analytics.sentiment_stability || 0) * 100).toFixed(0)}%
                  </p>
                </div>
                <CheckCircle className="w-8 h-8 text-blue-600" />
              </div>
            </div>
          </>
        )}
      </div>

      {/* Agent Distribution */}
      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <Activity className="w-4 h-4" />
          Agent Usage Distribution
        </h3>
        <div className="space-y-2">
          {Object.entries(stats.agent_distribution || {}).map(([agent, count]) => (
            <div key={agent} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {agentIcons[agent]}
                <span className="text-sm text-gray-700 capitalize">{agent}</span>
              </div>
              <span className="text-sm font-medium text-gray-900">{count}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Enhanced Sentiment Analysis */}
      {analytics && analytics.overall_sentiment && (
        <>
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Heart className="w-4 h-4" />
              Emotional State Analysis
            </h3>
            
            {/* Overall Sentiment Display */}
            <div className="flex items-center gap-4 mb-6 p-4 bg-gray-50 rounded-lg">
              <div className="w-16 h-16 rounded-full bg-primary-100 flex items-center justify-center text-2xl font-bold">
                {analytics.overall_sentiment === 'positive' && '😊'}
                {analytics.overall_sentiment === 'neutral' && '😐'}
                {analytics.overall_sentiment === 'stressed' && '😰'}
                {analytics.overall_sentiment === 'sad' && '😢'}
                {analytics.overall_sentiment === 'anxious' && '😟'}
                {analytics.overall_sentiment === 'frustrated' && '😤'}
              </div>
              <div className="flex-1">
                <p className="text-sm text-gray-600">Overall Emotional State</p>
                <p className="text-2xl font-bold text-gray-900 capitalize">
                  {analytics.overall_sentiment}
                </p>
                <p className="text-sm text-gray-500">
                  Confidence: {((analytics.sentiment_confidence || 0) * 100).toFixed(1)}% | 
                  Intensity: {analytics.emotional_intensity?.toFixed(1) || '0.0'}/10
                </p>
              </div>
            </div>

            {/* Enhanced Sentiment Distribution Chart */}
            {chartData.length > 0 && (
              <div className="space-y-4">
                <h4 className="font-medium text-gray-900">Sentiment Distribution</h4>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="sentiment" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} />
                    <Tooltip 
                      formatter={(value, name, props) => [
                        `${value} messages (${props.payload.percentage?.toFixed(1)}%)`,
                        'Count'
                      ]}
                      labelFormatter={(label) => `Sentiment: ${label}`}
                    />
                    <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Detailed Sentiment Breakdown */}
            {chartData.length > 0 && (
              <div className="mt-6">
                <h4 className="font-medium text-gray-900 mb-3">Detailed Breakdown</h4>
                <div className="space-y-2">
                  {chartData.map((item) => (
                    <div key={item.sentiment} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div 
                          className="w-4 h-4 rounded-full" 
                          style={{ backgroundColor: item.color }}
                        ></div>
                        <span className="font-medium capitalize">{item.sentiment}</span>
                      </div>
                      <div className="text-right text-sm">
                        <div className="font-semibold">{item.count} messages</div>
                        <div className="text-gray-500">
                          {item.percentage?.toFixed(1)}% | 
                          Conf: {(item.confidence * 100).toFixed(0)}% | 
                          Int: {item.intensity?.toFixed(1)}/10
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Emotional Intensity Analysis */}
          {intensityData.length > 0 && (
            <div className="card">
              <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Emotional Intensity Analysis
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={intensityData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="emotion" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip 
                    formatter={(value, name) => [`${value.toFixed(1)}/10`, 'Intensity']}
                    labelFormatter={(label) => `Emotion: ${label}`}
                  />
                  <Bar dataKey="intensity" fill="#f59e0b" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Confidence Metrics */}
          {confidenceData.length > 0 && (
            <div className="card">
              <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Shield className="w-4 h-4" />
                Analysis Reliability
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={confidenceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="metric" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip 
                    formatter={(value) => [`${value.toFixed(1)}%`, 'Confidence']}
                    labelFormatter={(label) => `Metric: ${label}`}
                  />
                  <Bar dataKey="value" fill="#10b981" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Support Level Assessment */}
          {analytics.needs_support && (
            <div className={`card ${
              analytics.needs_support.level === 'high'
                ? 'border-red-300 bg-red-50'
                : analytics.needs_support.level === 'moderate'
                ? 'border-yellow-300 bg-yellow-50'
                : 'border-green-300 bg-green-50'
            }`}>
              <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                {analytics.needs_support.level === 'high' ? (
                  <AlertTriangle className="w-4 h-4 text-red-600" />
                ) : analytics.needs_support.level === 'moderate' ? (
                  <AlertTriangle className="w-4 h-4 text-yellow-600" />
                ) : (
                  <CheckCircle className="w-4 h-4 text-green-600" />
                )}
                Support Level Assessment
              </h3>
              <div className="space-y-2">
                <p className={`font-medium ${
                  analytics.needs_support.level === 'high'
                    ? 'text-red-800'
                    : analytics.needs_support.level === 'moderate'
                    ? 'text-yellow-800'
                    : 'text-green-800'
                }`}>
                  {analytics.needs_support.level === 'high' && '🚨 HIGH - Immediate attention needed'}
                  {analytics.needs_support.level === 'moderate' && '⚠️ MODERATE - Monitor closely'}
                  {analytics.needs_support.level === 'low' && '✅ LOW - Managing well'}
                </p>
                {analytics.needs_support.recommendation && (
                  <p className="text-sm text-gray-700">{analytics.needs_support.recommendation}</p>
                )}
              </div>
            </div>
          )}

          {/* Detailed Insights */}
          {analytics.insights && analytics.insights.length > 0 && (
            <div className="card">
              <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Brain className="w-4 h-4" />
                Detailed Analysis Insights
              </h3>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {analytics.insights.map((insight, index) => (
                  <div key={index} className="text-sm text-gray-700 leading-relaxed">
                    {insight}
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default AnalyticsDashboard;