// spoofsniper_dataset_workflow.js
// Usage: include this in your landing page. Expects your Flask app to expose:
//  - POST /api/predict   (JSON { text, username, followers } -> { label, confidence, ... })
//  - POST /api/generate_report  (optional; returns dataset-level metrics)
// Place dataset CSV at: /static/data/dataset.csv (headers must include at least "text" and "label")

class SpoofSniperWorkflow {
    constructor(opts = {}) {
        this.isRunning = false;
        this.workflowSteps = [];
        this.currentStep = 0;
        this.results = [];

        // configuration
        this.predictEndpoint = opts.predictEndpoint || '/api/predict';
        this.reportEndpoint = opts.reportEndpoint || '/api/generate_report';
        this.datasetUrl = opts.datasetUrl || '/static/data/spoof_posts.csv';
        this.concurrentRequests = opts.concurrentRequests || 6; // concurrency for batch predict
        this.fallbackToLocalRules = opts.fallbackToLocalRules ?? true;

        // sample posts/accounts kept for legacy demo UI
        this.samplePosts = opts.samplePosts || [
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

        this.sampleAccounts = opts.sampleAccounts || [
            { username: "winner_now_123", followers: 12, following: 50, bio: "Win cash now!!!", profilePicDefault: true, created_days_ago: 5, recentPosts: 2, avgLikes: 0 },
            { username: "john_doe", followers: 1200, following: 300, bio: "Photographer & coffee lover", profilePicDefault: false, created_days_ago: 400, recentPosts: 150, avgLikes: 200 },
            { username: "save_money_today", followers: 45, following: 100, bio: "Free money and prizes", profilePicDefault: true, created_days_ago: 10, recentPosts: 1, avgLikes: 0 },
            { username: "naturelover", followers: 600, following: 80, bio: "Sunsets & hikes", profilePicDefault: false, created_days_ago: 700, recentPosts: 40, avgLikes: 60 }
        ];

        this.initWorkflow();
    }

    initWorkflow() {
        this.workflowSteps = [
            { name: 'Input Validation', action: 'validateInput' },
            { name: 'Text Preprocessing', action: 'preprocessText' },
            { name: 'Feature Extraction', action: 'extractFeatures' },
            { name: 'Account Analysis', action: 'analyzeAccount' },
            { name: 'Pattern Analysis', action: 'analyzePatterns' },
            { name: 'Remote ML Prediction', action: 'makeRemotePrediction' },
            { name: 'Confidence Scoring', action: 'calculateConfidence' },
            { name: 'Explanation Generation', action: 'generateExplanations' },
            { name: 'Result Formatting', action: 'formatResults' }
        ];
    }

    // ---------- Main workflow ----------
    async executeWorkflow(inputText, metadata = {}) {
        this.isRunning = true;
        this.results = [];
        this.currentStep = 0;

        let workflowData = {
            input: inputText,
            metadata,
            timestamp: new Date().toISOString()
        };

        try {
            for (let i = 0; i < this.workflowSteps.length; i++) {
                this.currentStep = i;
                const step = this.workflowSteps[i];
                // call action; if not found, skip gracefully
                if (typeof this[step.action] !== 'function') {
                    console.warn(`Action ${step.action} not implemented`);
                    continue;
                }
                const stepResult = await this[step.action](workflowData);
                this.results.push({
                    step: i + 1,
                    name: step.name,
                    action: step.action,
                    result: stepResult,
                    timestamp: new Date().toISOString()
                });
                workflowData = { ...workflowData, ...stepResult };
                // small visual delay
                await this.delay(150);
            }
            return workflowData;
        } catch (err) {
            console.error('Workflow error:', err);
            throw err;
        } finally {
            this.isRunning = false;
        }
    }

    // ---------- Individual steps ----------
    async validateInput(data) {
        const input = (data.input || '').toString();
        if (!input.trim()) throw new Error('Input text required');
        if (input.length > 5000) throw new Error('Input too long (max 5000 chars)');
        return { isValid: true, textLength: input.length, wordCount: input.split(/\s+/).filter(Boolean).length };
    }

    async preprocessText(data) {
        const original = (data.input || '').toString();
        const lowercase = original.toLowerCase();
        const cleaned = original.replace(/http\S+/gi, '').replace(/[^\w\s!?.,$]/g, '');
        const words = cleaned.split(/\s+/).filter(Boolean);
        const sentences = cleaned.split(/[.!?]+/).filter(s => s.trim().length > 0);
        return { processed: { original, lowercase, cleaned, words, sentences } };
    }

    async extractFeatures(data) {
        const p = data.processed;
        const wordCount = p.words.length || 1;
        const features = {
            length: p.original.length,
            wordCount,
            sentenceCount: p.sentences.length,
            avgWordLength: p.words.reduce((s, w) => s + w.length, 0) / wordCount,
            capsRatio: ((p.original.match(/[A-Z]/g) || []).length) / Math.max(p.original.length, 1),
            exclamationCount: (p.original.match(/!/g) || []).length,
            questionCount: (p.original.match(/\?/g) || []).length,
            digitCount: (p.original.match(/\d/g) || []).length,
            urgencyWords: this.countUrgencyWords(p.lowercase),
            moneyWords: this.countMoneyWords(p.lowercase),
            actionWords: this.countActionWords(p.lowercase),
            suspiciousUrls: this.countSuspiciousUrls(p.original)
        };
        return { features };
    }

    async analyzeAccount(data) {
        const account = data.metadata.account || {};
        const accountFeatures = {
            username: account.username || data.metadata.username || 'unknown',
            followers: account.followers || data.metadata.followers || 0,
            following: account.following || data.metadata.following || 0,
            followingRatio: (account.following || data.metadata.following || 1) === 0 ? 0 : ((account.followers || data.metadata.followers || 0) / (account.following || data.metadata.following || 1)),
            profilePicDefault: !!account.profilePicDefault,
            bioLength: (account.bio || '').length,
            bioSuspiciousWords: this.countBioSuspiciousWords((account.bio || '').toLowerCase()),
            createdDaysAgo: account.created_days_ago || data.metadata.account_age || 9999,
            isNewAccount: (account.created_days_ago || data.metadata.account_age || 9999) < 30,
            recentPosts: account.recentPosts || 0,
            avgLikes: account.avgLikes || 0
        };

        // simple rule scoring
        let score = 0;
        if (accountFeatures.profilePicDefault) score += 2;
        if (accountFeatures.isNewAccount) score += 2;
        if (accountFeatures.followers < 50) score += 1;
        if (accountFeatures.followingRatio < 0.1) score += 1;
        if (accountFeatures.bioSuspiciousWords) score += accountFeatures.bioSuspiciousWords * 2;
        score += this.usernameSuspiciousScore(accountFeatures.username);
        accountFeatures.suspicionScore = score;

        return { accountFeatures };
    }

    async analyzePatterns(data) {
        const { features, accountFeatures } = data;
        const patterns = {
            urgencyScore: features.urgencyWords * 2,
            moneyScore: features.moneyWords * 3,
            actionScore: features.actionWords * 2,
            capsScore: features.capsRatio > 0.3 ? 2 : 0,
            exclamationScore: features.exclamationCount > 2 ? 3 : 0,
            urlScore: features.suspiciousUrls * 4,
            accountScore: (accountFeatures && accountFeatures.suspicionScore) ? accountFeatures.suspicionScore : 0
        };
        patterns.totalSuspiciousScore = Object.values(patterns).reduce((s, v) => s + (typeof v === 'number' ? v : 0), 0);
        return { patterns };
    }

    // THIS is the key change: try the remote Flask model; fallback to local rules if needed
    async makeRemotePrediction(data) {
        const payload = {
            text: data.processed.cleaned || data.input,
            username: (data.accountFeatures && data.accountFeatures.username) || (data.metadata && data.metadata.username) || '',
            followers: (data.accountFeatures && data.accountFeatures.followers) || (data.metadata && data.metadata.followers) || 0
        };

        // attempt remote prediction
        try {
            const resp = await fetch(this.predictEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!resp.ok) {
                throw new Error(`Predict API returned ${resp.status}`);
            }
            const json = await resp.json();
            // Normalize possible label names
            const label = (json.label || json.prediction || '').toString().toLowerCase();
            const confidence = typeof json.confidence !== 'undefined' ? Number(json.confidence) : (json.confidenceScore || 0);

            return {
                remotePrediction: { label, confidence, raw: json },
                usedRemote: true
            };

        } catch (err) {
            console.warn('Remote predict failed:', err.message);
            // fallback: do the local rule-based prediction using combined patterns + account
            const local = await this.localRulePrediction(data);
            return {
                remotePrediction: { label: local.prediction, confidence: local.confidence, raw: { fallback: true } },
                usedRemote: false
            };
        }
    }

    async localRulePrediction(data) {
        const { patterns, accountFeatures } = data;
        const baseProbability = Math.min(patterns.totalSuspiciousScore / 25, 1);
        const accountBoost = Math.min((accountFeatures && accountFeatures.suspicionScore || 0) / 8, 1);
        let fakeProbability = Math.min(baseProbability + accountBoost * 0.6, 1);
        if (baseProbability > 0.6 && accountBoost > 0.4) fakeProbability = Math.min(fakeProbability + 0.15, 1);
        return { prediction: fakeProbability > 0.5 ? 'fake' : 'real', confidence: fakeProbability };
    }

    async calculateConfidence(data) {
        const label = data.remotePrediction && data.remotePrediction.label;
        const confidence = (data.remotePrediction && data.remotePrediction.confidence) || 0;
        const level = confidence > 0.8 ? 'high' : confidence > 0.6 ? 'medium' : 'low';
        return { finalLabel: label, confidence, confidenceLevel: level, confidencePct: Math.round(confidence * 100) };
    }

    async generateExplanations(data) {
        const explanations = [];
        const { patterns, features, accountFeatures } = data;

        if (patterns.urgencyScore > 0) explanations.push(`Urgency keywords: ${features.urgencyWords}`);
        if (patterns.moneyScore > 0) explanations.push(`Money/prize words: ${features.moneyWords}`);
        if (patterns.actionScore > 0) explanations.push(`Action words: ${features.actionWords}`);
        if (patterns.capsScore > 0) explanations.push(`High CAPS usage (${Math.round(features.capsRatio * 100)}%)`);
        if (patterns.exclamationScore > 0) explanations.push(`Multiple exclamation marks (${features.exclamationCount})`);
        if (patterns.urlScore > 0) explanations.push(`Suspicious URLs (${features.suspiciousUrls})`);
        if (accountFeatures && accountFeatures.suspicionScore > 0) {
            explanations.push(`Account suspicion score: ${accountFeatures.suspicionScore}`);
        }
        if (explanations.length === 0) explanations.push('No obvious red flags detected');

        return { explanations };
    }

    async formatResults(data) {
        const final = {
            prediction: data.finalLabel,
            confidence: data.confidencePct,
            explanations: data.explanations,
            features: data.features,
            patterns: data.patterns,
            accountFeatures: data.accountFeatures || null,
            timestamp: new Date().toISOString(),
            workflowVersion: 'dataset-1.0'
        };
        return { finalResult: final };
    }

    // ---------- Batch: run entire dataset through remote model ----------
    async runDatasetAnalysis(options = {}) {
        // options: limit (max rows), sample (boolean), concurrency override
        const limit = options.limit || null;
        const sample = options.sample || false;
        const concurrency = options.concurrency || this.concurrentRequests;

        // fetch dataset CSV from datasetUrl
        console.log('📥 Loading dataset from', this.datasetUrl);
        const csvText = await this.fetchText(this.datasetUrl);
        const rows = this.parseCsv(csvText);
        if (rows.length === 0) throw new Error('Dataset CSV parsed to 0 rows');

        // Expecting "text" and "label" columns
        if (!('text' in rows[0]) || !('label' in rows[0])) {
            throw new Error('Dataset must include "text" and "label" columns');
        }

        // optionally sample
        let toAnalyze = rows;
        if (sample && limit) {
            toAnalyze = this.shuffleArray(rows).slice(0, limit);
        } else if (limit) {
            toAnalyze = rows.slice(0, limit);
        }

        console.log(`🔎 Running dataset analysis on ${toAnalyze.length} rows (concurrency ${concurrency})`);

        // worker pool: batches of promises
        const results = [];
        let i = 0;
        const self = this;

        async function worker() {
            while (true) {
                let idx;
                // critical section
                if (i >= toAnalyze.length) return;
                idx = i++;
                const row = toAnalyze[idx];
                // run the same workflow steps but only those needed for predict
                try {
                    const minimalData = {
                        input: row.text,
                        metadata: {
                            username: row.username || '',
                            followers: row.followers ? Number(row.followers) : 0,
                            account: row.account || null
                        }
                    };
                    const processed = await self.preprocessText(minimalData);
                    const features = (await self.extractFeatures({ ...minimalData, ...processed })).features;
                    const accountFeatures = (await self.analyzeAccount({ ...minimalData })).accountFeatures;
                    const patterns = (await self.analyzePatterns({ features, accountFeatures })).patterns;
                    const predObj = await self.makeRemotePrediction({ processed, features, accountFeatures, metadata: minimalData.metadata, patterns });
                    // final confidence & label
                    const confidence = (await self.calculateConfidence({ remotePrediction: predObj.remotePrediction })).confidence;
                    const label = (await self.calculateConfidence({ remotePrediction: predObj.remotePrediction })).finalLabel;
                    results.push({ index: idx, expected: row.label.toString().toLowerCase(), predicted: (label || '').toString().toLowerCase(), confidence, rawRow: row });
                } catch (err) {
                    console.warn('Row error', idx, err.message);
                    results.push({ index: idx, expected: row.label, predicted: 'error', confidence: 0, error: err.message, rawRow: row });
                }
            }
        }

        // spawn workers
        const workers = [];
        for (let w = 0; w < concurrency; w++) workers.push(worker());
        await Promise.all(workers);

        // compute metrics
        const metrics = this.computeBatchMetrics(results);
        console.log('📊 Dataset analysis complete:', metrics);
        return { metrics, results };
    }

    computeBatchMetrics(results) {
        let correct = 0;
        let total = 0;
        const confusion = {}; // confusion[expected][predicted]
        for (const r of results) {
            if (r.predicted === 'error') continue;
            const expected = (r.expected || '').toString().toLowerCase();
            const predicted = (r.predicted || '').toString().toLowerCase();
            total++;
            if (expected === predicted) correct++;
            if (!confusion[expected]) confusion[expected] = {};
            confusion[expected][predicted] = (confusion[expected][predicted] || 0) + 1;
        }
        const accuracy = total ? (correct / total) : 0;
        return { total, correct, accuracy: Number(accuracy.toFixed(4)), confusion };
    }

    // ---------- Utilities ----------
    async fetchText(url) {
        const r = await fetch(url);
        if (!r.ok) throw new Error(`Failed to fetch ${url}: ${r.status}`);
        return await r.text();
    }

    // A tiny CSV parser: first row headers, simple CSV (no multiline fields)
    parseCsv(text) {
        const lines = text.split(/\r?\n/).filter(Boolean);
        if (lines.length === 0) return [];
        const headers = lines[0].split(',').map(h => h.trim());
        const rows = [];
        for (let i = 1; i < lines.length; i++) {
            const cols = this.splitCsvLine(lines[i]);
            if (cols.length === 0) continue;
            const obj = {};
            for (let j = 0; j < headers.length; j++) {
                obj[headers[j]] = (cols[j] !== undefined) ? cols[j].trim() : '';
            }
            rows.push(obj);
        }
        return rows;
    }

    // handle quoted commas
    splitCsvLine(line) {
        const out = [];
        let cur = '';
        let inQuotes = false;
        for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; continue; } // escaped
            if (ch === '"') { inQuotes = !inQuotes; continue; }
            if (ch === ',' && !inQuotes) { out.push(cur); cur = ''; continue; }
            cur += ch;
        }
        out.push(cur);
        return out;
    }

    shuffleArray(arr) {
        const a = arr.slice();
        for (let i = a.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [a[i], a[j]] = [a[j], a[i]];
        }
        return a;
    }

    // text helpers (copied/adapted from your original)
    countUrgencyWords(text) {
        const urgencyWords = ['urgent', 'emergency', 'act now', 'limited time', 'immediately', 'asap'];
        return urgencyWords.reduce((count, word) => {
            const regex = new RegExp(`\\b${this.escapeRegExp(word)}\\b`, 'gi');
            return count + (text.match(regex) || []).length;
        }, 0);
    }

    countMoneyWords(text) {
        const moneyWords = ['money', 'cash', 'prize', 'winner', 'free', 'dollar', 'win', '\\$'];
        return moneyWords.reduce((count, word) => {
            const regex = new RegExp(word.replace(/^\//,''), 'gi'); // word may contain backslash
            return count + (text.match(regex) || []).length;
        }, 0);
    }

    countActionWords(text) {
        const actionWords = ['click', 'verify', 'confirm', 'update', 'suspended', 'locked', 'hacked'];
        return actionWords.reduce((count, word) => {
            const regex = new RegExp(`\\b${this.escapeRegExp(word)}\\b`, 'gi');
            return count + (text.match(regex) || []).length;
        }, 0);
    }

    countSuspiciousUrls(text) {
        const urlRegex = /https?:\/\/[^\s]+/gi;
        const urls = text.match(urlRegex) || [];
        return urls.length;
    }

    countBioSuspiciousWords(text) {
        const suspicious = ['free money', 'win', 'prize', 'click here', 'dm for', 'gift', 'earn', 'income'];
        return suspicious.reduce((count, word) => {
            const regex = new RegExp(this.escapeRegExp(word), 'gi');
            return count + (text.match(regex) || []).length;
        }, 0);
    }

    usernameSuspiciousScore(username = '') {
        if (!username) return 0;
        let score = 0;
        const numbersCount = (username.match(/\d/g) || []).length;
        if (numbersCount > 3) score += 2;
        if (/(_now|winner|free|money|cash|prize|giveaway)/i.test(username)) score += 3;
        if (/^[\w\d]{2,6}\d{3,}$/i.test(username)) score += 1;
        if (/[_\-]{3,}/.test(username)) score += 1;
        return score;
    }

    escapeRegExp(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    delay(ms) {
        return new Promise(r => setTimeout(r, ms));
    }

    // ---------- Demo helpers ----------
    async startAutoDemo() {
        for (let i = 0; i < this.samplePosts.length; i++) {
            const post = this.samplePosts[i];
            const account = this.sampleAccounts[i % this.sampleAccounts.length];
            console.log(`\n--- Demo: Post #${i + 1} ---`);
            console.log('Post:', post);
            console.log('Account:', account.username, `followers:${account.followers}`, `created:${account.created_days_ago}`);
            try {
                const res = await this.executeWorkflow(post, { account });
                console.log('Result:', res.finalResult.prediction.toUpperCase(), `(${res.finalResult.confidence}% confidence)`);
                console.log('Explanations:', res.finalResult.explanations.join('; '));
            } catch (e) {
                console.error('Demo error:', e.message);
            }
            await this.delay(800);
        }
        console.log('Demo finished.');
    }

    // status / reset
    getStatus() {
        return {
            isRunning: this.isRunning,
            currentStep: this.currentStep,
            totalSteps: this.workflowSteps.length,
            lastResultsCount: this.results.length
        };
    }

    reset() {
        this.isRunning = false;
        this.currentStep = 0;
        this.results = [];
    }
}

// export for modules or browser global
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SpoofSniperWorkflow;
} else {
    window.SpoofSniperWorkflow = SpoofSniperWorkflow;
}
