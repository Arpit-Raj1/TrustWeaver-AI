
import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { AlertTriangle, Shield, TrendingUp, Users, Clock, MapPin } from "lucide-react";

const FraudMonitor = () => {
  const [liveData, setLiveData] = useState({
    activeThreats: 23,
    blockedTransactions: 156,
    suspiciousAccounts: 45,
    responseTime: "3.2s"
  });

  const [recentAlerts, setRecentAlerts] = useState([
    {
      id: 1,
      type: "Suspicious Network",
      description: "Multiple accounts sharing payment methods",
      riskScore: 94,
      timestamp: "2 min ago",
      status: "active",
      location: "US-West"
    },
    {
      id: 2,
      type: "Fraud Ring Detected",
      description: "Coordinated review manipulation",
      riskScore: 87,
      timestamp: "5 min ago", 
      status: "blocked",
      location: "EU-Central"
    },
    {
      id: 3,
      type: "Payment Anomaly",
      description: "Unusual transaction pattern detected",
      riskScore: 78,
      timestamp: "8 min ago",
      status: "investigating",
      location: "Asia-Pacific"
    }
  ]);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setLiveData(prev => ({
        ...prev,
        activeThreats: prev.activeThreats + Math.floor(Math.random() * 3) - 1,
        blockedTransactions: prev.blockedTransactions + Math.floor(Math.random() * 5),
        suspiciousAccounts: prev.suspiciousAccounts + Math.floor(Math.random() * 2) - 1,
        responseTime: `${(Math.random() * 2 + 2).toFixed(1)}s`
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const networkData = [
    { node: "Seller Network A", connections: 45, riskScore: 87, status: "high-risk" },
    { node: "Review Cluster B", connections: 23, riskScore: 72, status: "medium-risk" },
    { node: "Payment Group C", connections: 67, riskScore: 94, status: "blocked" },
    { node: "Account Ring D", connections: 12, riskScore: 56, status: "monitoring" }
  ];

  return (
    <div className="space-y-6">
      {/* Real-time Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Active Threats</p>
                <p className="text-2xl font-bold text-red-600">{liveData.activeThreats}</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-red-500" />
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Blocked Transactions</p>
                <p className="text-2xl font-bold text-orange-600">{liveData.blockedTransactions}</p>
              </div>
              <Shield className="h-8 w-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Suspicious Accounts</p>
                <p className="text-2xl font-bold text-yellow-600">{liveData.suspiciousAccounts}</p>
              </div>
              <Users className="h-8 w-8 text-yellow-500" />
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Response Time</p>
                <p className="text-2xl font-bold text-green-600">{liveData.responseTime}</p>
              </div>
              <Clock className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Live Alerts */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5 text-red-500" />
              <span>Live Fraud Alerts</span>
            </CardTitle>
            <CardDescription>
              Real-time detection via Graph Neural Networks and AWS Kinesis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentAlerts.map((alert) => (
                <div key={alert.id} className="p-4 border rounded-lg space-y-3">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center space-x-2 mb-1">
                        <h4 className="font-semibold text-sm">{alert.type}</h4>
                        <Badge 
                          variant={
                            alert.status === 'blocked' ? 'destructive' : 
                            alert.status === 'active' ? 'secondary' : 'outline'
                          }
                          className="text-xs"
                        >
                          {alert.status}
                        </Badge>
                      </div>
                      <p className="text-sm text-gray-600">{alert.description}</p>
                      <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                        <div className="flex items-center space-x-1">
                          <Clock className="h-3 w-3" />
                          <span>{alert.timestamp}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <MapPin className="h-3 w-3" />
                          <span>{alert.location}</span>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-600">Risk Score</p>
                      <p className="text-lg font-bold text-red-600">{alert.riskScore}%</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Progress value={alert.riskScore} className="flex-1 h-2" />
                    <span className="text-xs text-gray-500">{alert.riskScore}%</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Network Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5 text-blue-500" />
              <span>Fraud Network Analysis</span>
            </CardTitle>
            <CardDescription>
              Graph Neural Network detection of connected fraud patterns
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {networkData.map((network, index) => (
                <div key={index} className="p-3 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-semibold text-sm">{network.node}</h4>
                    <Badge 
                      variant={
                        network.status === 'blocked' ? 'destructive' :
                        network.status === 'high-risk' ? 'secondary' :
                        network.status === 'medium-risk' ? 'outline' : 'default'
                      }
                      className="text-xs"
                    >
                      {network.status}
                    </Badge>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-600">Connections</p>
                      <p className="font-semibold">{network.connections}</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Risk Score</p>
                      <p className="font-semibold text-red-600">{network.riskScore}%</p>
                    </div>
                  </div>
                  
                  <div className="mt-3">
                    <Progress value={network.riskScore} className="h-2" />
                  </div>
                </div>
              ))}
            </div>
            
            <div className="mt-4 p-3 bg-blue-50 rounded-lg">
              <div className="flex items-center space-x-2 text-sm text-blue-800">
                <TrendingUp className="h-4 w-4" />
                <span className="font-semibold">Network Insights</span>
              </div>
              <p className="text-sm text-blue-700 mt-1">
                Detected 3 new fraud clusters in the last hour. GNN analysis identified 
                suspicious payment sharing patterns across 67 accounts.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default FraudMonitor;
