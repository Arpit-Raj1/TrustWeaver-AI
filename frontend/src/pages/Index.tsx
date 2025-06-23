
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Shield, Eye, Brain, TrendingUp, AlertTriangle, CheckCircle, XCircle } from "lucide-react";
import ReviewAuthenticator from "@/components/ReviewAuthenticator";
import CounterfeitDetector from "@/components/CounterfeitDetector";
import FraudMonitor from "@/components/FraudMonitor";
import AnalyticsDashboard from "@/components/AnalyticsDashboard";

const Index = () => {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-white to-blue-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-orange-200/50 sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="bg-gradient-to-r from-orange-500 to-orange-600 p-2 rounded-xl">
                <Shield className="h-8 w-8 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-orange-600 to-blue-600 bg-clip-text text-transparent">
                  TrustWeaver AI
                </h1>
                <p className="text-sm text-gray-600">Amazon Marketplace Protection</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="secondary" className="bg-green-100 text-green-800">
                <CheckCircle className="h-3 w-3 mr-1" />
                System Active
              </Badge>
              <img src="https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg" alt="Amazon" className="h-8" />
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-6 py-12">
        <div className="text-center mb-12">
          <h2 className="text-5xl font-bold mb-4 bg-gradient-to-r from-orange-600 via-red-500 to-blue-600 bg-clip-text text-transparent">
            Safeguarding Amazon Marketplace
          </h2>
          <p className="text-xl text-gray-700 max-w-3xl mx-auto mb-8">
            Powered by Generative AI and AWS native services, TrustWeaver AI provides comprehensive protection 
            against fraudulent reviews, counterfeit products, and marketplace fraud.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
            <Card className="border-orange-200 hover:shadow-lg transition-shadow">
              <CardContent className="p-6 text-center">
                <Brain className="h-12 w-12 text-orange-500 mx-auto mb-4" />
                <h3 className="font-semibold text-lg mb-2">80%+ Accuracy</h3>
                <p className="text-sm text-gray-600">Fraudulent review detection using Bedrock LLMs</p>
              </CardContent>
            </Card>
            <Card className="border-blue-200 hover:shadow-lg transition-shadow">
              <CardContent className="p-6 text-center">
                <Eye className="h-12 w-12 text-blue-500 mx-auto mb-4" />
                <h3 className="font-semibold text-lg mb-2">99%+ Detection</h3>
                <p className="text-sm text-gray-600">Counterfeit product identification with Amazon Rekognition</p>
              </CardContent>
            </Card>
            <Card className="border-red-200 hover:shadow-lg transition-shadow">
              <CardContent className="p-6 text-center">
                <TrendingUp className="h-12 w-12 text-red-500 mx-auto mb-4" />
                <h3 className="font-semibold text-lg mb-2">5-Second Response</h3>
                <p className="text-sm text-gray-600">Real-time fraud detection with Graph Neural Networks</p>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Main Dashboard */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-8">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="reviews">Review Authentication</TabsTrigger>
            <TabsTrigger value="counterfeit">Counterfeit Detection</TabsTrigger>
            <TabsTrigger value="fraud">Fraud Monitor</TabsTrigger>
          </TabsList>

          <TabsContent value="overview">
            <AnalyticsDashboard />
          </TabsContent>

          <TabsContent value="reviews">
            <ReviewAuthenticator />
          </TabsContent>

          <TabsContent value="counterfeit">
            <CounterfeitDetector />
          </TabsContent>

          <TabsContent value="fraud">
            <FraudMonitor />
          </TabsContent>
        </Tabs>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-8 mt-16">
        <div className="container mx-auto px-6 text-center">
          <p className="text-gray-400">
            © 2024 TrustWeaver AI - Amazon HackOn Solution for Marketplace Protection
          </p>
          <div className="flex justify-center space-x-6 mt-4">
            <Badge variant="outline" className="text-gray-300 border-gray-600">
              AWS Bedrock
            </Badge>
            <Badge variant="outline" className="text-gray-300 border-gray-600">
              Amazon Rekognition
            </Badge>
            <Badge variant="outline" className="text-gray-300 border-gray-600">
              Amazon Kinesis
            </Badge>
            <Badge variant="outline" className="text-gray-300 border-gray-600">
              AWS Lambda
            </Badge>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
