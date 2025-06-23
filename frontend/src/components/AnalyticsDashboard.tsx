
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, TrendingDown, Shield, Eye, Brain, AlertTriangle } from "lucide-react";

const AnalyticsDashboard = () => {
  const weeklyData = [
    { day: 'Mon', reviews: 450, counterfeit: 23, fraud: 12 },
    { day: 'Tue', reviews: 520, counterfeit: 31, fraud: 18 },
    { day: 'Wed', reviews: 480, counterfeit: 19, fraud: 15 },
    { day: 'Thu', reviews: 610, counterfeit: 28, fraud: 22 },
    { day: 'Fri', reviews: 580, counterfeit: 35, fraud: 29 },
    { day: 'Sat', reviews: 720, counterfeit: 42, fraud: 31 },
    { day: 'Sun', reviews: 690, counterfeit: 38, fraud: 25 }
  ];

  const threatDistribution = [
    { name: 'Fake Reviews', value: 45, color: '#f97316' },
    { name: 'Counterfeit Products', value: 30, color: '#3b82f6' },
    { name: 'Payment Fraud', value: 25, color: '#ef4444' }
  ];

  const performanceMetrics = [
    {
      title: "Review Authentication",
      current: 87,
      target: 85,
      trend: "up",
      icon: Brain,
      color: "orange"
    },
    {
      title: "Counterfeit Detection",
      current: 94,
      target: 90,
      trend: "up", 
      icon: Eye,
      color: "blue"
    },
    {
      title: "Fraud Prevention",
      current: 91,
      target: 88,
      trend: "up",
      icon: Shield,
      color: "green"
    }
  ];

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {performanceMetrics.map((metric, index) => {
          const IconComponent = metric.icon;
          return (
            <Card key={index}>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium">{metric.title}</CardTitle>
                  <IconComponent className={`h-5 w-5 text-${metric.color}-500`} />
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-3xl font-bold">{metric.current}%</span>
                    <div className="flex items-center space-x-1">
                      {metric.trend === "up" ? (
                        <TrendingUp className="h-4 w-4 text-green-500" />
                      ) : (
                        <TrendingDown className="h-4 w-4 text-red-500" />
                      )}
                      <span className={`text-sm ${metric.trend === "up" ? "text-green-600" : "text-red-600"}`}>
                        +{metric.current - metric.target}%
                      </span>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm text-gray-600 mb-1">
                      <span>Target: {metric.target}%</span>
                      <span>Current: {metric.current}%</span>
                    </div>
                    <Progress value={metric.current} className="h-2" />
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Weekly Threat Analysis */}
        <Card>
          <CardHeader>
            <CardTitle>Weekly Threat Detection</CardTitle>
            <CardDescription>Detected threats across all protection systems</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={weeklyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="reviews" fill="#f97316" name="Fake Reviews" />
                <Bar dataKey="counterfeit" fill="#3b82f6" name="Counterfeit" />
                <Bar dataKey="fraud" fill="#ef4444" name="Fraud" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Threat Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Threat Distribution</CardTitle>
            <CardDescription>Breakdown of detected threats by category</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center">
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={threatDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={120}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {threatDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-1 gap-2 mt-4">
              {threatDistribution.map((item, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="text-sm">{item.name}</span>
                  </div>
                  <span className="text-sm font-semibold">{item.value}%</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* System Performance */}
      <Card>
        <CardHeader>
          <CardTitle>System Performance Overview</CardTitle>
          <CardDescription>Real-time monitoring of TrustWeaver AI components</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 bg-orange-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-semibold text-sm">Bedrock LLM</h4>
                <Badge variant="default" className="bg-green-100 text-green-800">
                  Online
                </Badge>
              </div>
              <p className="text-xs text-gray-600 mb-1">Processing Speed</p>
              <p className="text-lg font-bold">847ms avg</p>
            </div>
            
            <div className="p-4 bg-blue-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-semibold text-sm">Rekognition</h4>
                <Badge variant="default" className="bg-green-100 text-green-800">
                  Online
                </Badge>
              </div>
              <p className="text-xs text-gray-600 mb-1">Image Analysis</p>
              <p className="text-lg font-bold">1.2s avg</p>
            </div>
            
            <div className="p-4 bg-red-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-semibold text-sm">Kinesis Stream</h4>
                <Badge variant="default" className="bg-green-100 text-green-800">
                  Active
                </Badge>
              </div>
              <p className="text-xs text-gray-600 mb-1">Events/sec</p>
              <p className="text-lg font-bold">2.3k</p>
            </div>
            
            <div className="p-4 bg-green-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-semibold text-sm">Lambda Functions</h4>
                <Badge variant="default" className="bg-green-100 text-green-800">
                  Healthy
                </Badge>
              </div>
              <p className="text-xs text-gray-600 mb-1">Execution Time</p>
              <p className="text-lg font-bold">423ms avg</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default AnalyticsDashboard;
