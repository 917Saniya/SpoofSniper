// SpoofSniper Automation Workflow (n8n-style)
// This JavaScript file provides automated detection capabilities

class SpoofSniperWorkflow {
    constructor() {
        this.isRunning = false;
        this.workflowSteps = [];
        this.currentStep = 0;
        this.results = [];
        this.samplePosts = [
            "URGENT! Click here to claim your $1000 prize! Limited time offer!",
            "Congratulations! You've won $500! Verify your account now!",
            "Your account will be suspended! Click link to verify immediately!",
            "Just had a great day at the park with my family",
            "Looking forward to the weekend! Anyone have fun plans?",
            "Thanks everyone for the birthday wishes!",
            "FREE MONEY! Act now! Limited time offer! Click here!",
            "Emergency! Your account is hacked! Verify immediately!",
            "Beautiful sunset today. Nature never fails to amaze me",
            "Coffee and a good book - perfect morning"
        ];
        
        this.initWorkflow();
    }

    initWorkflow() {
        this.workflowSteps = [
            { name: 'Input Validation', action: 'validateInput' },
            { name: 'Text Preprocessing', action: 'preprocessText' },
            { name: 'Feature Extraction', action: 'extractFeatures' },
            { name: 'Pattern Analysis', action: 'analyzePatterns' },
            { name: 'ML Prediction', action: 'makePrediction' },
            { name: 'Confidence Scoring', action: 'calculateConfidence' },
            { name: 'Explanation Generation', action: 'generateExplanations' },
            { name: 'Result Formatting', action: 'formatResults' }
        ];
    }

    // Main workflow execution
    async executeWorkflow(inputText, metadata = {}) {
        console.log('🚀 Starting SpoofSniper Workflow...');
        this.isRunning = true;
        this.results = [];
        this.currentStep = 0;

        try {
            let workflowData = {
                input: inputText,
                metadata: metadata,
                timestamp: new Date().toISOString()
            };

            // Execute each step in sequence
            for (let i = 0; i < this.workflowSteps.length; i++) {
                this.currentStep = i;
                const step = this.workflowSteps[i];
                
                console.log(`📋 Step ${i + 1}: ${step.name}`);
                workflowData = await this.executeStep(step.action, workflowData);
                
                // Add delay for visual effect
                await this.delay(500);
            }

            console.log('✅ Workflow completed successfully!');
            return workflowData;

        } catch (error) {
            console.error('❌ Workflow failed:', error);
            throw error;
        } finally {
            this.isRunning = false;
        }
    }

    // Execute individual workflow step
    async executeStep(action, data) {
        const stepResult = await this[action](data);
        this.results.push({
            step: this.currentStep + 1,
            action: action,
            result: stepResult,
            timestamp: new Date().toISOString()
        });
        return { ...data, ...stepResult };
    }

    // Step 1: Input Validation
    async validateInput(data) {
        const { input } = data;
        
        if (!input || input.trim().length === 0) {
            throw new Error('Input text is required');
        }

        if (input.length > 1000) {
            throw new Error('Input text is too long (max 1000 characters)');
        }

        return {
            isValid: true,
            textLength: input.length,
            wordCount: input.split(' ').length
        };
    }

    // Step 2: Text Preprocessing
    async preprocessText(data) {
        const { input } = data;
        
        const processed = {
            original: input,
            lowercase: input.toLowerCase(),
            cleaned: input.replace(/[^\w\s!?.,$]/g, ''),
            words: input.split(/\s+/),
            sentences: input.split(/[.!?]+/).filter(s => s.trim().length > 0)
        };

        return { processed };
    }

    // Step 3: Feature Extraction
    async extractFeatures(data) {
        const { processed, metadata } = data;
        
        const features = {
            // Text features
            length: processed.original.length,
            wordCount: processed.words.length,
            sentenceCount: processed.sentences.length,
            avgWordLength: processed.words.reduce((sum, word) => sum + word.length, 0) / processed.words.length,
            
            // Character analysis
            capsRatio: (processed.original.match(/[A-Z]/g) || []).length / processed.original.length,
            exclamationCount: (processed.original.match(/!/g) || []).length,
            questionCount: (processed.original.match(/\?/g) || []).length,
            digitCount: (processed.original.match(/\d/g) || []).length,
            
            // Suspicious patterns
            urgencyWords: this.countUrgencyWords(processed.lowercase),
            moneyWords: this.countMoneyWords(processed.lowercase),
            actionWords: this.countActionWords(processed.lowercase),
            suspiciousUrls: this.countSuspiciousUrls(processed.original),
            
            // Account metadata
            followers: metadata.followers || 0,
            accountAge: metadata.account_age || 0,
            lowFollowers: (metadata.followers || 0) < 100,
            newAccount: (metadata.account_age || 0) < 30
        };

        return { features };
    }

    // Step 4: Pattern Analysis
    async analyzePatterns(data) {
        const { features } = data;
        
        const patterns = {
            urgencyScore: features.urgencyWords * 2,
            moneyScore: features.moneyWords * 3,
            actionScore: features.actionWords * 2,
            capsScore: features.capsRatio > 0.3 ? 2 : 0,
            exclamationScore: features.exclamationCount > 2 ? 3 : 0,
            urlScore: features.suspiciousUrls * 4,
            accountScore: (features.lowFollowers ? 1 : 0) + (features.newAccount ? 1 : 0),
            
            totalSuspiciousScore: 0
        };

        patterns.totalSuspiciousScore = Object.values(patterns).reduce((sum, score) => 
            typeof score === 'number' ? sum + score : sum, 0
        );

        return { patterns };
    }

    // Step 5: ML Prediction
    async makePrediction(data) {
        const { patterns, features } = data;
        
        // Simple rule-based prediction (can be replaced with actual ML model)
        const fakeProbability = Math.min(patterns.totalSuspiciousScore / 20, 1);
        const prediction = fakeProbability > 0.5 ? 'fake' : 'real';
        
        return {
            prediction,
            fakeProbability,
            realProbability: 1 - fakeProbability
        };
    }

    // Step 6: Confidence Scoring
    async calculateConfidence(data) {
        const { fakeProbability, realProbability } = data;
        
        const confidence = Math.max(fakeProbability, realProbability);
        const confidenceLevel = confidence > 0.8 ? 'high' : 
                               confidence > 0.6 ? 'medium' : 'low';

        return {
            confidence: confidence,
            confidenceLevel,
            confidencePercentage: Math.round(confidence * 100)
        };
    }

    // Step 7: Explanation Generation
    async generateExplanations(data) {
        const { features, patterns, prediction } = data;
        
        const explanations = [];
        
        if (patterns.urgencyScore > 0) {
            explanations.push(`Contains ${features.urgencyWords} urgency keywords`);
        }
        
        if (patterns.moneyScore > 0) {
            explanations.push(`Mentions money or prizes (${features.moneyWords} instances)`);
        }
        
        if (patterns.actionScore > 0) {
            explanations.push(`Contains suspicious action words (${features.actionWords} instances)`);
        }
        
        if (patterns.capsScore > 0) {
            explanations.push(`High use of capital letters (${Math.round(features.capsRatio * 100)}%)`);
        }
        
        if (patterns.exclamationScore > 0) {
            explanations.push(`Multiple exclamation marks (${features.exclamationCount})`);
        }
        
        if (patterns.urlScore > 0) {
            explanations.push(`Contains suspicious URLs (${features.suspiciousUrls})`);
        }
        
        if (patterns.accountScore > 0) {
            if (features.lowFollowers) explanations.push('Low follower count');
            if (features.newAccount) explanations.push('New account');
        }
        
        if (explanations.length === 0) {
            explanations.push('No obvious red flags detected');
        }

        return { explanations };
    }

    // Step 8: Result Formatting
    async formatResults(data) {
        const { prediction, confidencePercentage, explanations, features, patterns } = data;
        
        return {
            finalResult: {
                prediction: prediction,
                confidence: confidencePercentage,
                explanations: explanations,
                features: features,
                patterns: patterns,
                timestamp: new Date().toISOString(),
                workflowVersion: '1.0.0'
            }
        };
    }

    // Helper methods
    countUrgencyWords(text) {
        const urgencyWords = ['urgent', 'emergency', 'act now', 'limited time', 'immediately', 'asap'];
        return urgencyWords.reduce((count, word) => {
            const regex = new RegExp(`\\b${word}\\b`, 'gi');
            return count + (text.match(regex) || []).length;
        }, 0);
    }

    countMoneyWords(text) {
        const moneyWords = ['money', 'cash', 'prize', 'winner', 'free', 'dollar', '$', 'win'];
        return moneyWords.reduce((count, word) => {
            const regex = new RegExp(`\\b${word}\\b`, 'gi');
            return count + (text.match(regex) || []).length;
        }, 0);
    }

    countActionWords(text) {
        const actionWords = ['click', 'verify', 'confirm', 'update', 'suspended', 'locked', 'hacked'];
        return actionWords.reduce((count, word) => {
            const regex = new RegExp(`\\b${word}\\b`, 'gi');
            return count + (text.match(regex) || []).length;
        }, 0);
    }

    countSuspiciousUrls(text) {
        const urlRegex = /https?:\/\/[^\s]+/gi;
        const urls = text.match(urlRegex) || [];
        return urls.length;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Automated demo mode
    async startAutoDemo() {
        console.log('🤖 Starting automated demo...');
        
        for (let i = 0; i < this.samplePosts.length; i++) {
            const post = this.samplePosts[i];
            console.log(`\n📝 Analyzing post ${i + 1}: "${post}"`);
            
            try {
                const result = await this.executeWorkflow(post, {
                    followers: Math.floor(Math.random() * 1000),
                    account_age: Math.floor(Math.random() * 365)
                });
                
                console.log(`✅ Result: ${result.finalResult.prediction.toUpperCase()} (${result.finalResult.confidence}% confidence)`);
                console.log(`📋 Explanations: ${result.finalResult.explanations.join(', ')}`);
                
            } catch (error) {
                console.error(`❌ Error analyzing post ${i + 1}:`, error.message);
            }
            
            // Delay between posts
            await this.delay(2000);
        }
        
        console.log('\n🎉 Automated demo completed!');
    }

    // Get workflow status
    getStatus() {
        return {
            isRunning: this.isRunning,
            currentStep: this.currentStep,
            totalSteps: this.workflowSteps.length,
            results: this.results
        };
    }

    // Reset workflow
    reset() {
        this.isRunning = false;
        this.currentStep = 0;
        this.results = [];
    }
}

// Initialize workflow instance
const spoofSniperWorkflow = new SpoofSniperWorkflow();

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SpoofSniperWorkflow;
} else {
    window.SpoofSniperWorkflow = SpoofSniperWorkflow;
    window.spoofSniperWorkflow = spoofSniperWorkflow;
}

// Auto-start demo if on landing page
document.addEventListener('DOMContentLoaded', function() {
    if (window.location.pathname.includes('landing')) {
        console.log('🎯 Landing page detected - Workflow ready!');
        
        // Add workflow controls to page
        const workflowControls = document.createElement('div');
        workflowControls.innerHTML = `
            <div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;">
                <button id="start-workflow-demo" class="btn btn-primary">
                    <i class="fas fa-robot"></i> Start Auto Demo
                </button>
                <button id="workflow-status" class="btn btn-info">
                    <i class="fas fa-info"></i> Status
                </button>
            </div>
        `;
        document.body.appendChild(workflowControls);
        
        // Add event listeners
        document.getElementById('start-workflow-demo').addEventListener('click', function() {
            spoofSniperWorkflow.startAutoDemo();
        });
        
        document.getElementById('workflow-status').addEventListener('click', function() {
            const status = spoofSniperWorkflow.getStatus();
            alert(`Workflow Status:\nRunning: ${status.isRunning}\nStep: ${status.currentStep}/${status.totalSteps}\nResults: ${status.results.length}`);
        });
    }
});

