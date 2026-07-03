/* ML Animations — section viz + SFT/LoRA gallery */
(function () {
    'use strict';

    function observeOnce(el, fn) {
        if (!el) return;
        const obs = new IntersectionObserver(e => {
            if (e[0].isIntersecting) { fn(); obs.disconnect(); }
        }, { threshold: 0.15 });
        obs.observe(el);
    }

    /** Bind canvas once — logical size stays in data-* (never read canvas.width for layout). */
    function bindCanvas(canvas) {
        if (!canvas) return null;

        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        if (!canvas.dataset.logicalW) {
            const fromData = parseInt(canvas.getAttribute('data-logical-w'), 10);
            const fromAttr = parseInt(canvas.getAttribute('width'), 10);
            const w = fromData || (fromAttr > 0 && fromAttr <= 512 ? fromAttr : 220);
            const fromDataH = parseInt(canvas.getAttribute('data-logical-h'), 10);
            const fromAttrH = parseInt(canvas.getAttribute('height'), 10);
            const h = fromDataH || (fromAttrH > 0 && fromAttrH <= 512 ? fromAttrH : 88);
            canvas.dataset.logicalW = String(w);
            canvas.dataset.logicalH = String(h);
        }

        const W = parseInt(canvas.dataset.logicalW, 10);
        const H = parseInt(canvas.dataset.logicalH, 10);
        const boundDpr = canvas.dataset.boundDpr;

        if (boundDpr !== String(dpr) || !canvas.dataset.canvasReady) {
            canvas.width = W * dpr;
            canvas.height = H * dpr;
            canvas.style.width = W + 'px';
            canvas.style.height = H + 'px';
            canvas.dataset.boundDpr = String(dpr);
            canvas.dataset.canvasReady = '1';
        }

        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        return { ctx, W, H, dpr };
    }

    function cosineBetas(n) {
        const out = [];
        for (let t = 0; t < n; t++) {
            const v = Math.cos(((t / n) + 0.008) / 1.008 * Math.PI / 2);
            out.push(Math.min(0.999, 1 - v * v));
        }
        return out;
    }

    function buildAlphaBars(betas) {
        const alphas = betas.map(b => 1 - b);
        return alphas.reduce((acc, a, i) => {
            acc.push(i === 0 ? a : acc[i - 1] * a);
            return acc;
        }, []);
    }

    function randn() {
        const u = 1 - Math.random();
        const v = Math.random();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    function softmax(arr) {
        const m = Math.max(...arr);
        const ex = arr.map(v => Math.exp(v - m));
        const s = ex.reduce((a, b) => a + b, 0);
        return ex.map(v => v / s);
    }

    function lerp(a, b, t) { return a + (b - a) * t; }

    function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

    function register(replayReg, key, start) {
        if (replayReg && key) replayReg[key] = start;
    }

    function ceGrad(logits, yStar) {
        const p = softmax(logits);
        return p.map((pi, i) => pi - (i === yStar ? 1 : 0));
    }

    function initProjectsSFT(replayReg) {
        const canvas = document.getElementById('viz-projects');
        const captionEl = document.getElementById('projects-caption');
        if (!canvas) return;
        const { ctx, W, H } = bindCanvas(canvas);

        const tokens = ['fix', 'code', 'QLoRA', 'eval'];
        const yStar = 2;
        const eta = 0.22;
        const logitsInit = [-0.35, 0.55, 0.05, 0.25];
        let logits, displayLogits, step, rafId, lastTime;

        function reset() {
            logits = logitsInit.slice();
            displayLogits = logitsInit.slice();
            step = 0;
            lastTime = 0;
        }

        function draw() {
            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = 'rgba(0,0,0,0.35)';
            ctx.fillRect(0, 0, W, H);

            const probs = softmax(displayLogits);
            const loss = -Math.log(Math.max(probs[yStar], 1e-8));

            ctx.font = '600 7px JetBrains Mono';
            ctx.fillStyle = '#41c7df';
            ctx.fillText('task: fine-tune project checkpoint', 6, 10);

            const n = tokens.length;
            const bw = (W - 16) / n;
            const baseY = H - 18;
            tokens.forEach((tok, i) => {
                const x = 8 + i * bw;
                const h = probs[i] * 36;
                const isY = i === yStar;
                const g = ctx.createLinearGradient(0, baseY - h, 0, baseY);
                g.addColorStop(0, isY ? '#d0ff00' : '#e94560');
                g.addColorStop(1, isY ? 'rgba(208,255,0,0.15)' : 'rgba(233,69,96,0.12)');
                ctx.fillStyle = g;
                ctx.fillRect(x + 1, baseY - h, bw - 4, h);
                ctx.strokeStyle = isY ? '#d0ff00' : 'rgba(255,255,255,0.15)';
                ctx.lineWidth = isY ? 1.2 : 1;
                ctx.strokeRect(x + 1.5, baseY - h + 0.5, bw - 5, h);
                ctx.font = '600 6px JetBrains Mono';
                ctx.fillStyle = isY ? '#d0ff00' : '#bbb';
                ctx.textAlign = 'center';
                ctx.fillText(tok, x + bw / 2, baseY + 9);
                ctx.font = '5px JetBrains Mono';
                ctx.fillStyle = '#888';
                ctx.fillText(`p=${probs[i].toFixed(2)}`, x + bw / 2, baseY - h - 2);
                ctx.textAlign = 'left';
            });

            ctx.font = '600 7px JetBrains Mono';
            ctx.fillStyle = loss < 0.35 ? '#d0ff00' : '#ffe25f';
            ctx.fillText(`L = −log p(y*|x) = ${loss.toFixed(3)}`, 6, 24);
            ctx.fillStyle = 'rgba(255,255,255,0.55)';
            ctx.font = '6px JetBrains Mono';
            ctx.fillText(`y* = "${tokens[yStar]}"  ·  ∇_z L = p − 𝟙[y*]  ·  step ${step}`, 6, 36);

            if (captionEl) {
                captionEl.textContent = step === 0
                    ? 'SFT · −log p(y*|x)'
                    : `SFT · p(${tokens[yStar]}) = ${probs[yStar].toFixed(3)}`;
            }
        }

        function tick(now) {
            if (now - lastTime > 85) {
                const grad = ceGrad(logits, yStar);
                logits = logits.map((z, i) => z - eta * grad[i]);
                step = (step + 1) % 24;
                if (step === 0) logits = logitsInit.slice();
                lastTime = now;
            }
            displayLogits = displayLogits.map((v, i) => lerp(v, logits[i], 0.14));
            draw();
            rafId = requestAnimationFrame(tick);
        }

        function start() {
            if (rafId) cancelAnimationFrame(rafId);
            reset();
            lastTime = performance.now();
            if (captionEl) captionEl.textContent = 'SFT · −log p(y*|x)';
            rafId = requestAnimationFrame(tick);
        }

        register(replayReg, 'projects', start);
        observeOnce(canvas.closest('.section-viz'), start);
        reset();
        draw();
    }

    function initPublicationActivations(replayReg) {
        const canvas = document.getElementById('viz-publication');
        const captionEl = document.getElementById('pub-caption');
        if (!canvas) return;
        const { ctx, W, H } = bindCanvas(canvas);

        const relu = x => Math.max(0, x);
        const gelu = x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
        const sigmoid = x => 1 / (1 + Math.exp(-x));
        const fns = [
            { fn: relu, c: '#e94560', label: 'ReLU', yScale: 1.15 },
            { fn: gelu, c: '#ffe25f', label: 'GELU', yScale: 1.15 },
            { fn: sigmoid, c: '#41c7df', label: 'σ(x)', yScale: 1.0 }
        ];
        let t, rafId, xSmooth;

        function crispLine(ctx) {
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';
        }

        function draw() {
            ctx.clearRect(0, 0, W, H);
            const ox = 34, oy = H - 22, scX = 28, scY = 26;
            const xTarget = -1.4 + (Math.sin(t * 0.018) + 1) * 1.55;
            xSmooth = xSmooth == null ? xTarget : lerp(xSmooth, xTarget, 0.06);

            ctx.fillStyle = 'rgba(0,0,0,0.35)';
            ctx.fillRect(0, 0, W, H);

            for (let gy = 0; gy <= 4; gy++) {
                const y = oy - (gy / 4) * scY * 1.15;
                ctx.strokeStyle = 'rgba(255,255,255,0.06)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(ox, Math.round(y) + 0.5);
                ctx.lineTo(W - 8, Math.round(y) + 0.5);
                ctx.stroke();
            }

            ctx.strokeStyle = 'rgba(255,255,255,0.22)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(Math.round(ox) + 0.5, 10);
            ctx.lineTo(Math.round(ox) + 0.5, Math.round(oy) + 0.5);
            ctx.lineTo(W - 8, Math.round(oy) + 0.5);
            ctx.stroke();

            fns.forEach(({ fn, c, label, yScale }, fi) => {
                ctx.beginPath();
                crispLine(ctx);
                for (let px = 0; px <= W - ox - 12; px++) {
                    const x = px / scX - 1.8;
                    const y = oy - fn(x) * scY * yScale;
                    const sx = ox + px;
                    if (px === 0) ctx.moveTo(sx, y);
                    else ctx.lineTo(sx, y);
                }
                ctx.strokeStyle = c;
                ctx.lineWidth = 2.2;
                ctx.stroke();

                ctx.font = '600 9px JetBrains Mono';
                ctx.fillStyle = c;
                ctx.fillText(label, ox + fi * 72, 14);
            });

            fns.forEach(({ fn, c, yScale }, fi) => {
                const yOut = fn(xSmooth) * scY * yScale;
                const px = ox + (xSmooth + 1.8) * scX;
                ctx.beginPath();
                ctx.arc(px, oy - yOut, 3.5, 0, Math.PI * 2);
                ctx.fillStyle = c;
                ctx.globalAlpha = fi === 0 ? 1 : 0.35;
                ctx.fill();
                ctx.globalAlpha = 1;
            });

            ctx.setLineDash([3, 3]);
            ctx.strokeStyle = 'rgba(208,255,0,0.35)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(ox + (xSmooth + 1.8) * scX, oy);
            ctx.lineTo(ox + (xSmooth + 1.8) * scX, oy - relu(xSmooth) * scY * 1.15);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.font = '600 8px JetBrains Mono';
            ctx.fillStyle = 'rgba(255,255,255,0.75)';
            ctx.fillText(`x = ${xSmooth.toFixed(2)}`, 8, H - 6);
            ctx.fillStyle = '#d0ff00';
            ctx.fillText(`ReLU(x) = ${relu(xSmooth).toFixed(2)}`, 88, H - 6);
            if (captionEl) captionEl.textContent = 'ReLU · GELU · σ(x)';
        }

        function tick() { t++; draw(); rafId = requestAnimationFrame(tick); }

        function start() {
            if (rafId) cancelAnimationFrame(rafId);
            t = 0;
            xSmooth = null;
            rafId = requestAnimationFrame(tick);
        }

        register(replayReg, 'publication', start);
        observeOnce(canvas.closest('.section-viz'), start);
        draw();
    }

    function initSlimDiffCard(replayReg) {
        const canvas = document.getElementById('slim-diff-viz');
        if (!canvas) return;
        const { ctx, W, H } = bindCanvas(canvas);

        const T = 32;
        const alphaBars = buildAlphaBars(cosineBetas(T));
        const stack = [
            { label: 'def twoSum(nums, target):', type: 'ctx' },
            { label: '    lo, hi = 0, len(nums)-1', type: 'ctx' },
            { label: '    while lo > hi:  # wrong', type: 'bad' },
            { label: '    return lo, hi', type: 'ctx' }
        ];
        const fixed = '    while lo < hi:';
        const maskIdx = 2;
        let tStep, phase, lastTime, rafId, flash;

        function reset() {
            tStep = T;
            phase = 'repair';
            lastTime = 0;
            flash = 0;
        }

        function corruptText(progress) {
            if (progress > 0.82) {
                const n = Math.ceil(fixed.length * ((progress - 0.82) / 0.18));
                return fixed.slice(0, n) + (n < fixed.length ? '▌' : '');
            }
            const noise = ['>', '<', '?', '#', '▒', '░'];
            const base = '    while lo > hi:';
            return base.split('').map((ch, i) => {
                if (i < 4) return ch;
                return progress < 0.35 && i % 3 === 0
                    ? noise[Math.floor((1 - progress) * noise.length)] : ch;
            }).join('') + '  # corrupt';
        }

        function draw() {
            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = 'rgba(0,0,0,0.45)';
            ctx.fillRect(0, 0, W, H);

            const progress = phase === 'done' ? 1 : 1 - tStep / T;
            const lh = 12;
            const startY = 5;

            stack.forEach((line, i) => {
                const y = startY + i * lh;
                let text = line.label;
                let color = 'rgba(255,255,255,0.42)';
                let bg = null;

                if (i === maskIdx) {
                    if (phase === 'done' || progress > 0.92) {
                        text = fixed;
                        color = '#d0ff00';
                        bg = `rgba(208,255,0,${0.12 + flash * 0.12})`;
                    } else {
                        text = corruptText(progress);
                        color = '#ffe25f';
                        bg = 'rgba(233,69,96,0.22)';
                    }
                }

                if (bg) {
                    ctx.fillStyle = bg;
                    ctx.fillRect(3, y - 8, W - 6, lh);
                }
                if (i === maskIdx && phase !== 'done' && progress < 0.92) {
                    ctx.font = '600 6px JetBrains Mono';
                    ctx.fillStyle = '#e94560';
                    ctx.fillText('[M]', W - 22, y);
                }
                ctx.font = '600 6.5px JetBrains Mono';
                ctx.fillStyle = color;
                ctx.textAlign = 'left';
                const clip = text.length > 34 ? text.slice(0, 33) + '…' : text;
                ctx.fillText(clip, 6, y);
            });

            ctx.font = '6px JetBrains Mono';
            ctx.fillStyle = 'rgba(255,255,255,0.45)';
            const ab = tStep > 0 ? alphaBars[tStep - 1] : 1;
            ctx.fillText(
                phase === 'done' ? '✓ line canonical · strict match' : `t=${tStep}  ᾱ=${ab.toFixed(2)}  denoise [M]`,
                5, H - 2
            );
        }

        function tick(now) {
            if (phase === 'repair' && now - lastTime > 65) {
                if (tStep > 0) tStep--;
                else phase = 'done';
                lastTime = now;
            }
            if (phase === 'done') flash = Math.min(1, flash + 0.05);
            draw();
            rafId = requestAnimationFrame(tick);
        }

        function start() {
            if (rafId) cancelAnimationFrame(rafId);
            reset();
            lastTime = performance.now();
            rafId = requestAnimationFrame(tick);
        }

        register(replayReg, 'slim-diff', start);
        observeOnce(canvas.closest('.project-inline-viz'), start);
        reset();
        draw();
    }

    function initAwardsLoRA(replayReg) {
        const canvas = document.getElementById('viz-awards');
        const captionEl = document.getElementById('awards-caption');
        if (!canvas) return;
        const { ctx, W, H } = bindCanvas(canvas);

        const rank = 2;
        const d = 6;
        let rafId, startTime;
        const CYCLE = 8000;

        function reset() { startTime = performance.now(); }

        function seg(t, a, b) {
            return easeOut(Math.min(1, Math.max(0, (t - a) / (b - a))));
        }

        function drawMiniMat(ctx, x, y, cols, rows, cs, color, alpha, pulse) {
            ctx.globalAlpha = alpha;
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const v = 0.35 + 0.65 * Math.abs(Math.sin(c * 1.6 + r * 2.2 + pulse));
                    ctx.fillStyle = color;
                    ctx.globalAlpha = alpha * v;
                    ctx.fillRect(x + c * cs, y + r * cs, cs - 0.8, cs - 0.8);
                }
            }
            ctx.globalAlpha = 1;
        }

        function drawFlow(ctx, x1, y1, x2, y2, prog, color) {
            for (let i = 0; i < 4; i++) {
                const f = (prog + i * 0.22) % 1;
                ctx.globalAlpha = Math.sin(f * Math.PI) * 0.85;
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(lerp(x1, x2, f), lerp(y1, y2, f), 1.8, 0, Math.PI * 2);
                ctx.fill();
            }
            ctx.globalAlpha = 1;
        }

        function draw() {
            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = 'rgba(0,0,0,0.35)';
            ctx.fillRect(0, 0, W, H);

            const t = ((performance.now() - startTime) % CYCLE) / CYCLE;
            const pulse = performance.now() * 0.003;
            const pA = seg(t, 0.06, 0.28);
            const pMul = seg(t, 0.24, 0.48);
            const pMerge = seg(t, 0.46, 0.78);
            const cs = 4;

            drawMiniMat(ctx, 6, 28, d, d, cs, '#41c7df', 0.85, pulse);
            ctx.font = '600 6px JetBrains Mono';
            ctx.fillStyle = '#41c7df';
            ctx.fillText('W', 6, 24);
            ctx.fillStyle = 'rgba(65,199,223,0.7)';
            ctx.fillText('d×d', 6, 58);

            if (pA > 0) {
                drawMiniMat(ctx, 38, 32, rank, d, cs, '#ffe25f', 0.35 + pA * 0.6, pulse);
                drawMiniMat(ctx, 50, 28, d, rank, cs, '#ffe25f', 0.35 + pA * 0.6, pulse);
                ctx.fillStyle = '#ffe25f';
                ctx.fillText('A', 38, 28);
                ctx.fillText('B', 50, 24);
                ctx.font = '5px JetBrains Mono';
                ctx.fillText(`r=${rank}`, 38, 60);
            }

            if (pMul > 0.05) {
                ctx.strokeStyle = `rgba(255,226,95,${0.3 + pMul * 0.55})`;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(32, 40);
                ctx.lineTo(38, 40);
                ctx.moveTo(48, 36);
                ctx.lineTo(50, 36);
                ctx.stroke();
                ctx.font = '600 6px JetBrains Mono';
                ctx.fillStyle = '#ffe25f';
                ctx.fillText('B·A', 40, 36);
                drawFlow(ctx, 32, 40, 38, 40, pMul, '#ffe25f');
            }

            if (pMerge > 0) {
                const mx = 118;
                const my = 28;
                const glow = 0.15 + Math.sin(pulse) * 0.08;
                ctx.fillStyle = `rgba(208,255,0,${glow + pMerge * 0.3})`;
                ctx.fillRect(mx - 2, my - 2, d * cs + 6, d * cs + 6);
                drawMiniMat(ctx, mx, my, d, d, cs, '#d0ff00', 0.5 + pMerge * 0.5, pulse);
                ctx.font = '600 7px JetBrains Mono';
                ctx.fillStyle = '#d0ff00';
                ctx.fillText("W′", mx + 4, 24);
                drawFlow(ctx, 74, 40, mx, 40, pMerge, '#d0ff00');
            }

            ctx.font = '600 6px JetBrains Mono';
            ctx.fillStyle = 'rgba(255,255,255,0.75)';
            ctx.fillText("W′ = W + B·A", 6, 10);
            ctx.font = '5px JetBrains Mono';
            ctx.fillStyle = 'rgba(255,255,255,0.45)';
            ctx.fillText(`rank r=${rank} ≪ d=${d}  ·  train A,B only`, 6, H - 4);

            const caps = ['W frozen', 'init A, B', 'B·A low-rank', 'W′ = W + BA', 'cert checkpoint ✓'];
            const ci = Math.min(4, Math.floor(t * 5));
            if (captionEl) captionEl.textContent = caps[ci];
        }

        function tick() {
            draw();
            rafId = requestAnimationFrame(tick);
        }

        function start() {
            if (rafId) cancelAnimationFrame(rafId);
            reset();
            if (captionEl) captionEl.textContent = 'LoRA · W′ = W + BA';
            rafId = requestAnimationFrame(tick);
        }

        register(replayReg, 'awards', start);
        observeOnce(canvas.closest('.section-viz'), start);
        reset();
        draw();
    }

    window.MLAnimations = {
        bindCanvas,
        initSection(replayReg) {
            initProjectsSFT(replayReg);
            initPublicationActivations(replayReg);
            initSlimDiffCard(replayReg);
            initAwardsLoRA(replayReg);
        }
    };
})();
