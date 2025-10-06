import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { TrendingUp, MessageSquare, Brain, Heart, BookOpen, Activity } from 'lucide-react';

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

const AnalyticsDashboard = ({ stats, analytics }) => {
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    if (analytics?.sentiment_distribution) {
      const data = Object.entries(analytics.sentiment_distribution).map(([key, value]) => ({
        sentiment: key,
        count: value,
        color: sentimentColors[key] || '#6b7280',
      }));
      setChartData(data);
    }
  }, [analytics]);

  if (!stats || stats.total_messages === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 text-center">
        <TrendingUp className="w-12 h-12 text-gray-400 mb-4" />
        <p className="text-gray-600">
          Start a conversation to see analytics
        </p>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          Conversation Analytics
        </h2>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 gap-4">
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
      </div>

      {/* Agent Distribution */}
      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-3">Agent Usage</h3>
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

      {/* Sentiment Analysis */}
      {analytics && analytics.overall_sentiment && (
        <>
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-3">Emotional Tone</h3>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 rounded-full bg-primary-100 flex items-center justify-center text-2xl">
                {analytics.overall_sentiment === 'positive' && '😊'}
                {analytics.overall_sentiment === 'neutral' && '😐'}
                {analytics.overall_sentiment === 'stressed' && '😰'}
                {analytics.overall_sentiment === 'sad' && '😢'}
                {analytics.overall_sentiment === 'anxious' && '😟'}
                {analytics.overall_sentiment === 'frustrated' && '😤'}
              </div>
              <div>
                <p className="text-sm text-gray-600">Overall Sentiment</p>
                <p className="text-lg font-semibold text-gray-900 capitalize">
                  {analytics.overall_sentiment}
                </p>
              </div>
            </div>

            {chartData.length > 0 && (
              <div className="mt-4">
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="sentiment" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Support Level */}
          {analytics.needs_support && (
            <div className={`card ${
              analytics.needs_support.level === 'high'
                ? 'border-red-300 bg-red-50'
                : analytics.needs_support.level === 'moderate'
                ? 'border-yellow-300 bg-yellow-50'
                : 'border-green-300 bg-green-50'
            }`}>
              <h3 className="font-semibold text-gray-900 mb-2">Support Level</h3>
              {analytics.needs_support.level === 'high' && (
                <p className="text-sm text-red-800">
                  ⚠️ High stress detected - Extra support recommended
                </p>
              )}
              {analytics.needs_support.level === 'moderate' && (
                <p className="text-sm text-yellow-800">
                  💛 Some stress detected - Monitor closely
                </p>
              )}
              {analytics.needs_support.level === 'low' && (
                <p className="text-sm text-green-800">
                  ✓ Conversation appears positive
                </p>
              )}
              {analytics.needs_support.recommendation && (
                <p className="text-sm text-gray-700 mt-2">
                  {analytics.needs_support.recommendation}
                </p>
              )}
            </div>
          )}

          {/* Insights */}
          {analytics.insights && analytics.insights.length > 0 && (
            <div className="card">
              <h3 className="font-semibold text-gray-900 mb-3">Insights</h3>
              <ul className="space-y-2">
                {analytics.insights.map((insight, index) => (
                  <li key={index} className="text-sm text-gray-700 flex items-start gap-2">
                    <span className="text-primary-600 mt-0.5">•</span>
                    <span>{insight}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default AnalyticsDashboard;
