/* ═══════════════════════════════════════════════════════════
   Pragadeesh KMS — Portfolio Engine
   Hero: step-by-step dense NN forward pass → profile output
   ═══════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    // ─────────────────────────────────────────────────────────
    // HERO NEURAL NETWORK — sequential forward pass animation
    // ─────────────────────────────────────────────────────────
    function initHeroNetwork() {
        const canvas = document.getElementById('hero-nn-canvas');
        const profile = document.getElementById('heroProfile');
        const stageLabel = document.getElementById('nnStageLabel');
        const phaseLabel = document.getElementById('heroPhaseLabel');
        const replayBtn = document.getElementById('replayNN');
        const lossVal = document.getElementById('lossVal');
        const lossEpoch = document.getElementById('lossEpoch');

        if (!canvas || !profile) return;

        const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        const ctx = canvas.getContext('2d');

        const INPUT_LABELS = [
            'Python', 'PyTorch', 'Diffusion', 'GPT',
            'RAG', 'RLHF', 'Training', 'Transformers'
        ];
        const LAYER_COUNTS = [8, 12, 18, 10, 1];
        const LAYER_NAMES = [
            'input · job requirements',
            'hidden₁ · feature extraction',
            'hidden₂ · dense representation',
            'hidden₃ · decision boundary',
            'output · candidate match'
        ];

        let W = 0, H = 0, dpr = 1;
        let lastPrep = '';
        let layers = [];
        let edges = [];
        let visibleLayers = 0;
        let visibleInputs = 0;
        let activeEdges = new Set();
        let litNodes = new Set();
        let pulses = [];
        let phase = 'idle';
        let phaseStart = 0;
        let animId = null;
        let profileShown = false;
        let replayCount = 0;

        function speedMult() {
            return Math.min(1 + replayCount * 0.08, 1.75);
        }

        function buildTopology() {
            layers = [];
            edges = [];
            const inputX = Math.max(108, W * 0.11);
            const outputX = W - Math.max(108, W * 0.12);
            const padTop = 0.035;
            const padBottom = 0.035;
            const usableH = H * (1 - padTop - padBottom);
            const startY = H * padTop;

            const layerX = LAYER_COUNTS.map((_, i) =>
                inputX + ((outputX - inputX) / (LAYER_COUNTS.length - 1)) * i
            );

            LAYER_COUNTS.forEach((count, li) => {
                const nodes = [];
                const gap = usableH / (count + 1);
                for (let ni = 0; ni < count; ni++) {
                    nodes.push({
                        id: `${li}-${ni}`,
                        x: layerX[li],
                        y: startY + gap * (ni + 1),
                        layer: li,
                        label: li === 0 ? INPUT_LABELS[ni] : null,
                        alpha: 0,
                        scale: 0,
                        glow: 0
                    });
                }
                layers.push(nodes);
            });

            for (let li = 0; li < layers.length - 1; li++) {
                layers[li].forEach(a => {
                    layers[li + 1].forEach(b => {
                        edges.push({ from: a, to: b, li, alpha: 0, key: `${a.id}->${b.id}` });
                    });
                });
            }
        }

        function positionProfileAtOutput() {
            if (!layers.length) return;
            const out = layers[layers.length - 1][0];
            const pct = Math.min(Math.max(((out.x + 4) / W) * 100, 72), 90);
            profile.style.left = `${pct}%`;
            profile.style.top = `${(out.y / H) * 100}%`;
            profile.style.right = 'auto';
        }

        function resize() {
            const stage = canvas.parentElement;
            if (!stage) return;

            const style = getComputedStyle(stage);
            const padT = parseFloat(style.paddingTop) || 0;
            const padB = parseFloat(style.paddingBottom) || 0;
            const padL = parseFloat(style.paddingLeft) || 0;
            const padR = parseFloat(style.paddingRight) || 0;

            dpr = Math.min(window.devicePixelRatio || 1, 2);
            W = Math.max(1, Math.floor(stage.clientWidth - padL - padR));
            H = Math.max(1, Math.floor(stage.clientHeight - padT - padB - 24));

            const prepKey = `${W}x${H}@${dpr}`;
            if (prepKey === lastPrep) return;
            lastPrep = prepKey;

            canvas.width = W * dpr;
            canvas.height = H * dpr;
            canvas.style.width = W + 'px';
            canvas.style.height = H + 'px';
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            buildTopology();
            positionProfileAtOutput();
        }

        function setLabel(text) {
            if (stageLabel) stageLabel.textContent = text;
        }

        function setPhase(text) {
            if (phaseLabel) phaseLabel.textContent = text;
        }

        function spawnPulses(fromLayer, count, baseSpeed) {
            const fromNodes = layers[fromLayer];
            const toNodes = layers[fromLayer + 1];
            if (!fromNodes || !toNodes) return;
            const sm = speedMult();

            for (let i = 0; i < count; i++) {
                const from = fromNodes[Math.floor(Math.random() * fromNodes.length)];
                const to = toNodes[Math.floor(Math.random() * toNodes.length)];
                pulses.push({
                    from, to,
                    t: Math.random() * 0.12,
                    speed: (baseSpeed + Math.random() * 0.004) * sm,
                    li: fromLayer
                });
            }
        }

        function showProfile() {
            if (profileShown) return;
            profileShown = true;
            phase = 'done';
            profile.removeAttribute('aria-hidden');
            profile.classList.add('visible');
            positionProfileAtOutput();
            setPhase('forward pass complete · σ(z) > τ');
            setLabel('output · Pragadeesh KMS');
            if (replayBtn) replayBtn.hidden = false;
        }

        function reset() {
            visibleLayers = 0;
            visibleInputs = 0;
            activeEdges.clear();
            litNodes.clear();
            pulses = [];
            phase = 'idle';
            profileShown = false;
            profile.classList.remove('visible');
            profile.setAttribute('aria-hidden', 'true');
            if (replayBtn) replayBtn.hidden = true;
            layers.forEach(layer => layer.forEach(n => { n.alpha = 0; n.scale = 0; n.glow = 0; }));
            edges.forEach(e => { e.alpha = 0; });
        }

        function startSequence() {
            reset();
            if (reduced) {
                visibleLayers = LAYER_COUNTS.length;
                visibleInputs = INPUT_LABELS.length;
                layers.flat().forEach(n => { n.alpha = 1; n.scale = 1; });
                edges.forEach(e => { e.alpha = 0.12; });
                litNodes = new Set(layers.flat().map(n => n.id));
                showProfile();
                draw();
                return;
            }
            phase = 'inputs';
            phaseStart = performance.now();
            setPhase('loading input tensor · job requirements');
            setLabel(LAYER_NAMES[0]);
            animate();
        }

        function updatePhase(now) {
            const elapsed = now - phaseStart;
            const sm = speedMult();
            const inputDelay = 380;
            const edgeDuration = 650;
            const pulseWait = 1400;
            const pulseTimeout = 4000;

            if (phase === 'inputs') {
                const idx = Math.min(Math.floor(elapsed / inputDelay), INPUT_LABELS.length);
                if (idx !== visibleInputs) {
                    visibleInputs = idx;
                    for (let i = 0; i < visibleInputs; i++) {
                        const n = layers[0][i];
                        n.alpha = 1;
                        n.scale = 1;
                    }
                }
                if (visibleInputs >= INPUT_LABELS.length && elapsed > INPUT_LABELS.length * inputDelay + 400) {
                    phase = 'edges-0';
                    phaseStart = now;
                    setPhase('W₁ initialized · dense connectivity');
                }
            }

            else if (phase.startsWith('edges-')) {
                const li = parseInt(phase.split('-')[1], 10);
                const progress = Math.min(elapsed / edgeDuration, 1);
                const ease = 1 - Math.pow(1 - progress, 2);
                const nextLayer = layers[li + 1];

                edges.filter(e => e.li === li).forEach(e => { e.alpha = ease * 0.14; });

                visibleLayers = li + 2;
                if (nextLayer) {
                    nextLayer.forEach(n => {
                        n.alpha = ease;
                        n.scale = 0.35 + ease * 0.65;
                        n.glow = ease * 0.4;
                    });
                }

                if (progress >= 1) {
                    if (nextLayer) {
                        nextLayer.forEach(n => {
                            n.alpha = 1;
                            n.scale = 1;
                            n.glow = 0.55;
                        });
                    }
                    phase = `pulse-${li}`;
                    phaseStart = now;
                    setPhase(`forward pass L${li + 1} → L${li + 2} · ReLU(z)`);
                    setLabel(LAYER_NAMES[li + 1] || LAYER_NAMES[LAYER_NAMES.length - 1]);
                    spawnPulses(li, li === 0 ? 18 : li === 1 ? 30 : 24, 0.0064 - li * 0.0009);
                }
            }

            else if (phase.startsWith('pulse-')) {
                const li = parseInt(phase.split('-')[1], 10);
                const layerDone = pulses.filter(p => p.li === li).length === 0 && elapsed > pulseWait;

                if (layerDone || (elapsed > pulseTimeout && pulses.filter(p => p.li === li).length < 3)) {
                    layers[li + 1].forEach(n => {
                        n.alpha = 1;
                        n.scale = 1;
                        n.glow = 1;
                        litNodes.add(n.id);
                    });
                    visibleLayers = li + 2;

                    if (li < layers.length - 2) {
                        phase = `edges-${li + 1}`;
                        phaseStart = now;
                        setPhase(`activating hidden_${li + 2} · ${LAYER_COUNTS[li + 2]} units`);
                    } else {
                        phase = 'output';
                        phaseStart = now;
                        setPhase('softmax · candidate selection');
                        setLabel(LAYER_NAMES[4]);
                    }
                }
            }

            else if (phase === 'output') {
                const out = layers[layers.length - 1][0];
                if (!profileShown) {
                    out.alpha = 1;
                    out.scale = 1;
                    out.glow = 1 + Math.sin(elapsed * 0.006) * 0.3;
                    litNodes.add(out.id);
                }
                if (elapsed > 900) showProfile();
            }
        }

        function draw() {
            ctx.clearRect(0, 0, W, H);

            // edges
            edges.forEach(e => {
                if (e.alpha <= 0) return;
                ctx.beginPath();
                ctx.moveTo(e.from.x, e.from.y);
                ctx.lineTo(e.to.x, e.to.y);
                const g = ctx.createLinearGradient(e.from.x, e.from.y, e.to.x, e.to.y);
                g.addColorStop(0, `rgba(255, 226, 95, ${e.alpha})`);
                g.addColorStop(1, `rgba(233, 69, 96, ${e.alpha})`);
                ctx.strokeStyle = g;
                ctx.lineWidth = 0.6;
                ctx.stroke();
            });

            // active edge highlight during pulses
            pulses.forEach(p => {
                const a = 0.35 + Math.sin(p.t * Math.PI) * 0.2;
                ctx.beginPath();
                ctx.moveTo(p.from.x, p.from.y);
                ctx.lineTo(p.to.x, p.to.y);
                ctx.strokeStyle = `rgba(65, 199, 223, ${a})`;
                ctx.lineWidth = 1.2;
                ctx.stroke();
            });

            // nodes
            layers.forEach((layer, li) => {
                if (li >= visibleLayers && li > 0) return;
                const isOutput = li === layers.length - 1;
                if (isOutput && profileShown) return;

                layer.forEach((n, ni) => {
                    if (li === 0 && ni >= visibleInputs) return;
                    if (n.alpha <= 0.01) return;

                    const r = li === 0 ? 9 : isOutput ? 11 : 6 + li * 0.65;
                    const glowR = r + 9 + n.glow * 7;

                    ctx.beginPath();
                    ctx.arc(n.x, n.y, glowR, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(233, 69, 96, ${0.06 + n.glow * 0.08})`;
                    ctx.fill();

                    ctx.beginPath();
                    ctx.arc(n.x, n.y, r * n.scale, 0, Math.PI * 2);
                    const ng = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r);
                    if (li === 0) {
                        ng.addColorStop(0, '#ffe25f');
                        ng.addColorStop(1, '#e94560');
                    } else if (isOutput) {
                        ng.addColorStop(0, '#41c7df');
                        ng.addColorStop(1, '#e94560');
                    } else {
                        ng.addColorStop(0, '#41c7df');
                        ng.addColorStop(1, '#8b5cf6');
                    }
                    ctx.fillStyle = ng;
                    ctx.globalAlpha = n.alpha;
                    ctx.fill();
                    ctx.globalAlpha = 1;

                    if (n.label && n.alpha > 0.5) {
                        const labelX = n.x - r - 14;
                        ctx.font = '600 11px JetBrains Mono, monospace';
                        ctx.fillStyle = 'rgba(255, 226, 95, 0.95)';
                        ctx.textAlign = 'right';
                        ctx.fillText(n.label, labelX, n.y + 4);
                    }
                });
            });

            // pulse dots
            pulses.forEach(p => {
                const x = p.from.x + (p.to.x - p.from.x) * p.t;
                const y = p.from.y + (p.to.y - p.from.y) * p.t;

                ctx.beginPath();
                ctx.arc(x, y, 5, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(208, 255, 0, 0.22)';
                ctx.fill();

                ctx.beginPath();
                ctx.arc(x, y, 4, 0, Math.PI * 2);
                ctx.fillStyle = '#d0ff00';
                ctx.shadowColor = '#d0ff00';
                ctx.shadowBlur = 18;
                ctx.fill();
                ctx.shadowBlur = 0;
            });
        }

        function animate(now) {
            if (phase !== 'idle' && phase !== 'done') updatePhase(now || performance.now());

            pulses = pulses.filter(p => {
                p.t += p.speed;
                if (p.t >= 1) {
                    p.to.glow = Math.min(p.to.glow + 0.35, 1);
                    litNodes.add(p.to.id);
                }
                return p.t < 1;
            });

            draw();
            animId = requestAnimationFrame(animate);
        }

        // Loss curve ticker synced with NN
        function initLossTicker() {
            if (!lossVal || !lossEpoch) return;
            let loss = 2.847;
            let epoch = 0;
            setInterval(() => {
                if (!profileShown) {
                    loss = Math.max(0.041, loss * (0.92 + Math.random() * 0.04));
                    epoch = Math.min(99, epoch + 1);
                } else {
                    loss = 0.041;
                    epoch = 100;
                }
                lossVal.textContent = loss.toFixed(3);
                lossEpoch.textContent = epoch;
            }, 180);
        }

        replayBtn?.addEventListener('click', () => {
            replayCount++;
            startSequence();
        });
        function onResize() {
            resize();
            draw();
        }

        const stageEl = canvas.parentElement;
        if (stageEl && typeof ResizeObserver !== 'undefined') {
            const ro = new ResizeObserver(onResize);
            ro.observe(stageEl);
        }
        window.addEventListener('resize', onResize);
        if (window.visualViewport) {
            window.visualViewport.addEventListener('resize', onResize);
            window.visualViewport.addEventListener('scroll', onResize);
        }

        onResize();
        startSequence();
        initLossTicker();
    }

    // ── Cursor glow ──
    function initCursorGlow() {
        const glow = document.querySelector('.cursor-glow');
        if (!glow) return;
        document.addEventListener('mousemove', e => {
            glow.style.left = e.clientX + 'px';
            glow.style.top = e.clientY + 'px';
        });
    }

    // ── Header ──
    function initHeader() {
        const header = document.querySelector('.site-header');
        const toggle = document.querySelector('.nav-toggle');
        const nav = document.querySelector('.main-nav');

        window.addEventListener('scroll', () => {
            header?.classList.toggle('scrolled', window.scrollY > 40);
        }, { passive: true });

        toggle?.addEventListener('click', () => {
            const open = nav.classList.toggle('open');
            toggle.setAttribute('aria-expanded', open);
        });

    }

    // ── Navigation ──
    function initNavigation() {
        const nav = document.querySelector('.main-nav');
        const header = document.querySelector('.site-header');
        if (!nav) return;

        const sections = [...document.querySelectorAll('section[id]')]
            .filter(s => s.id && s.id !== 'hero');
        const sectionIds = new Set(sections.map(s => s.id));
        let scrollLockUntil = 0;

        function headerOffset() {
            return header ? header.offsetHeight : 72;
        }

        function setActive(id) {
            nav.querySelectorAll('.nav-link').forEach(link => {
                const on = link.getAttribute('href') === `#${id}`;
                link.classList.toggle('active', on);
                if (on) link.setAttribute('aria-current', 'page');
                else link.removeAttribute('aria-current');
            });
        }

        function scrollToSection(id) {
            const target = document.getElementById(id);
            if (!target) return;
            const top = target.getBoundingClientRect().top + window.scrollY - headerOffset() + 1;
            window.scrollTo({ top, behavior: 'smooth' });
        }

        function syncActiveFromScroll() {
            if (performance.now() < scrollLockUntil || !sections.length) return;
            const probe = window.scrollY + headerOffset() + 48;
            let current = sections[0].id;
            sections.forEach(section => {
                const top = section.getBoundingClientRect().top + window.scrollY;
                if (probe >= top) current = section.id;
            });
            setActive(current);
        }

        nav.addEventListener('click', e => {
            const link = e.target.closest('a.nav-link');
            if (!link) return;
            e.preventDefault();
            const id = link.getAttribute('href')?.slice(1);
            if (!id || !sectionIds.has(id)) return;
            scrollLockUntil = performance.now() + 1000;
            setActive(id);
            if (history.replaceState) history.replaceState(null, '', `#${id}`);
            else location.hash = id;
            scrollToSection(id);
            nav.classList.remove('open');
        });

        window.addEventListener('scroll', syncActiveFromScroll, { passive: true });
        window.addEventListener('resize', syncActiveFromScroll, { passive: true });
        window.addEventListener('hashchange', () => {
            const id = location.hash.slice(1);
            if (sectionIds.has(id)) {
                scrollLockUntil = performance.now() + 1000;
                setActive(id);
                scrollToSection(id);
            }
        });

        const hashId = location.hash.slice(1);
        if (hashId && sectionIds.has(hashId)) setActive(hashId);
        else syncActiveFromScroll();
    }

    // ── Scroll reveal ──
    function initScrollReveal() {
        document.querySelectorAll('.reveal').forEach(el => {
            const obs = new IntersectionObserver(entries => {
                entries.forEach(e => {
                    if (e.isIntersecting) { e.target.classList.add('revealed'); obs.unobserve(e.target); }
                });
            }, { threshold: 0.08 });
            obs.observe(el);
        });
    }

    // ── Experience timeline ──
    function initExperienceTimeline() {
        /* cards activate via gradient descent viz */
    }

    // ── Project filters ──
    function initProjectFilters() {
        const btns = document.querySelectorAll('.filter-btn');
        const cards = document.querySelectorAll('.project-card[data-category]');
        btns.forEach(btn => {
            btn.addEventListener('click', () => {
                btns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                cards.forEach(c => {
                    c.classList.toggle('hidden', btn.dataset.filter !== 'all' && c.dataset.category !== btn.dataset.filter);
                });
            });
        });
    }

    // ── Skills ──
    const skillsData = {
        languages: [
            { name: 'Python', icon: 'fab fa-python' }, { name: 'HTML', icon: 'fab fa-html5' },
            { name: 'CSS', icon: 'fab fa-css3-alt' }, { name: 'JavaScript', icon: 'fab fa-js' },
            { name: 'C++', icon: 'fas fa-code' }, { name: 'Node.js', icon: 'fab fa-node-js' },
            { name: 'SQL', icon: 'fas fa-database' }, { name: 'TypeScript', icon: 'fas fa-file-code' },
            { name: 'React', icon: 'fab fa-react' }
        ],
        expertise: [
            { name: 'Deep Learning', icon: 'fas fa-brain' }, { name: 'Machine Learning', icon: 'fas fa-robot' },
            { name: 'NLP', icon: 'fas fa-language' }, { name: 'Diffusion Models', icon: 'fas fa-cloud' },
            { name: 'Generative AI', icon: 'fas fa-magic' }, { name: 'Reinforcement Learning', icon: 'fas fa-sync-alt' },
            { name: 'Computer Vision', icon: 'fas fa-eye' }, { name: 'Prompt Engineering', icon: 'fas fa-keyboard' }
        ],
        platforms: [
            { name: 'OpenAI Playground', icon: 'fas fa-play-circle' }, { name: 'Microsoft Azure', icon: 'fab fa-microsoft' },
            { name: 'AWS', icon: 'fab fa-aws' }, { name: 'Google Cloud', icon: 'fab fa-google' },
            { name: 'Hugging Face', icon: 'fas fa-face-smile' }, { name: 'Vertex AI', icon: 'fas fa-cube' },
            { name: 'Kaggle', icon: 'fab fa-kaggle' }, { name: 'Docker', icon: 'fab fa-docker' }
        ],
        libraries: [
            { name: 'PyTorch', icon: 'fas fa-fire' }, { name: 'TensorFlow', icon: 'fas fa-project-diagram' },
            { name: 'Keras', icon: 'fas fa-layer-group' }, { name: 'Scikit-learn', icon: 'fas fa-chart-line' },
            { name: 'NumPy', icon: 'fas fa-calculator' }, { name: 'Pandas', icon: 'fas fa-table' },
            { name: 'Transformers', icon: 'fas fa-exchange-alt' }, { name: 'OpenCV', icon: 'fas fa-camera' },
            { name: 'Matplotlib', icon: 'fas fa-chart-bar' }
        ]
    };

    function initSkills() {
        const cats = document.querySelectorAll('.skill-category');
        const display = document.getElementById('skillsDisplay');
        if (!display) return;

        function show(cat) {
            display.innerHTML = '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';
            setTimeout(() => {
                display.innerHTML = '';
                skillsData[cat].forEach(s => {
                    const el = document.createElement('div');
                    el.className = 'skill-item';
                    el.innerHTML = `<div class="skill-icon"><i class="${s.icon}"></i></div><div class="skill-name">${s.name}</div>`;
                    display.appendChild(el);
                });
            }, 900);
        }

        cats.forEach(c => c.addEventListener('click', () => {
            cats.forEach(x => x.classList.remove('active'));
            c.classList.add('active');
            show(c.dataset.category);
        }));
        show('languages');
    }

    // ── Viz replay registry ──
    const vizReplay = {};

    function bindReplayButtons() {
        document.querySelectorAll('[data-replay]').forEach(btn => {
            btn.addEventListener('click', e => {
                e.stopPropagation();
                vizReplay[btn.dataset.replay]?.();
            });
        });
    }

    function observeViz(el, onStart) {
        let fired = false;
        const obs = new IntersectionObserver(entries => {
            if (entries[0].isIntersecting && !fired) {
                fired = true;
                onStart();
                obs.disconnect();
            }
        }, { threshold: 0.25 });
        obs.observe(el);
        return () => { fired = false; onStart(); };
    }

    // ── About: Conv2d 3×3 scan → pixel grid "ABOUT ME" ──
    function initAboutCNN() {
        const canvas = document.getElementById('viz-about');
        const captionEl = document.getElementById('about-caption');
        if (!canvas || !window.MLAnimations) return;

        const bound = MLAnimations.bindCanvas(canvas);
        if (!bound) return;
        const { ctx, W, H } = bound;

        const GLYPHS = {
            A: ['.#.', '#.#', '###', '#.#', '#.#', '#.#', '...'],
            B: ['##.', '#.#', '##.', '#.#', '#.#', '##.', '...'],
            O: ['.#.', '#.#', '#.#', '#.#', '#.#', '.#.', '...'],
            U: ['#.#', '#.#', '#.#', '#.#', '#.#', '###', '...'],
            T: ['###', '.#.', '.#.', '.#.', '.#.', '.#.', '...'],
            M: ['#.#', '###', '#.#', '#.#', '#.#', '#.#', '...'],
            E: ['###', '#..', '##.', '#..', '#..', '###', '...'],
            ' ': ['..', '..', '..', '..', '..', '..', '..']
        };
        const COLORS = {
            A: '#e94560', B: '#ff7a59', O: '#ffe25f', U: '#a8f5b4',
            T: '#41c7df', M: '#8b5cf6', E: '#d0ff00'
        };
        const TEXT = 'ABOUT ME';
        const ROWS = 7;
        const KERNEL = [
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9]
        ];

        function buildTarget() {
            const cols = TEXT.split('').reduce((n, ch) => n + (ch === ' ' ? 2 : 3), 0);
            const target = Array.from({ length: ROWS }, () => Array(cols).fill(null));
            const binary = Array.from({ length: ROWS }, () => Array(cols).fill(0));
            let cx = 0;
            TEXT.split('').forEach(ch => {
                const g = GLYPHS[ch];
                const w = ch === ' ' ? 2 : 3;
                if (g) {
                    for (let r = 0; r < ROWS; r++) {
                        for (let c = 0; c < w; c++) {
                            if (g[r] && g[r][c] === '#') {
                                target[r][cx + c] = COLORS[ch] || '#ffffff';
                                binary[r][cx + c] = 1;
                            }
                        }
                    }
                }
                cx += w;
            });
            return { target, binary, cols };
        }

        const { target, binary, cols } = buildTarget();
        const scanOrder = [];
        for (let y = 0; y < ROWS; y++) {
            for (let x = 0; x < cols; x++) scanOrder.push({ y, x });
        }

        let padded, reveal, noise, scanIdx, phase, lastTime, rafId;
        const stepMs = 38;

        function resetState() {
            padded = Array.from({ length: ROWS + 2 }, () => Array(cols + 2).fill(0));
            for (let y = 0; y < ROWS; y++) {
                for (let x = 0; x < cols; x++) {
                    padded[y + 1][x + 1] = binary[y][x] ? 1 : Math.random() * 0.3;
                }
            }
            reveal = Array.from({ length: ROWS }, () => Array(cols).fill(0));
            noise = Array.from({ length: ROWS }, () =>
                Array.from({ length: cols }, () => Math.random())
            );
            scanIdx = 0;
            phase = 'scan';
            lastTime = 0;
        }

        function convAt(outY, outX) {
            let sum = 0;
            for (let ky = 0; ky < 3; ky++) {
                for (let kx = 0; kx < 3; kx++) {
                    sum += KERNEL[ky][kx] * padded[outY + ky][outX + kx];
                }
            }
            return sum;
        }

        function draw() {
            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = 'rgba(0,0,0,0.35)';
            ctx.fillRect(0, 0, W, H);

            const padX = 10;
            const padY = 8;
            const cellW = (W - padX * 2) / cols;
            const cellH = (H - padY * 2) / ROWS;
            const gridW = cellW * cols;
            const gridH = cellH * ROWS;

            ctx.strokeStyle = 'rgba(65, 199, 223, 0.3)';
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 2]);
            ctx.strokeRect(padX - cellW, padY - cellH, gridW + 2 * cellW, gridH + 2 * cellH);
            ctx.setLineDash([]);

            for (let y = 0; y < ROWS; y++) {
                for (let x = 0; x < cols; x++) {
                    const px = padX + x * cellW;
                    const py = padY + y * cellH;
                    const t = target[y][x];
                    const rev = reveal[y][x];
                    const blur = 1 - rev;
                    let r, g, b;
                    if (t) {
                        r = parseInt(t.slice(1, 3), 16);
                        g = parseInt(t.slice(3, 5), 16);
                        b = parseInt(t.slice(5, 7), 16);
                    } else {
                        r = 16; g = 20; b = 34;
                    }
                    const nr = Math.floor(r * rev + (32 + noise[y][x] * 70) * blur);
                    const ng = Math.floor(g * rev + (28 + noise[y][x] * 45) * blur);
                    const nb = Math.floor(b * rev + (42 + noise[y][x] * 55) * blur);
                    ctx.fillStyle = `rgb(${nr},${ng},${nb})`;
                    ctx.fillRect(px + 0.5, py + 0.5, cellW - 1.5, cellH - 1.5);
                    if (t && rev > 0.85) {
                        ctx.fillStyle = `rgba(255,255,255,${(rev - 0.85) * 0.35})`;
                        ctx.fillRect(px + 1, py + 1, cellW - 2.5, cellH - 2.5);
                    }
                }
            }

            if (phase === 'scan' && scanIdx < scanOrder.length) {
                const { y, x } = scanOrder[scanIdx];
                const kx0 = padX + x * cellW - cellW;
                const ky0 = padY + y * cellH - cellH;
                ctx.strokeStyle = 'rgba(255, 226, 95, 0.95)';
                ctx.lineWidth = 1.5;
                ctx.strokeRect(kx0 + 0.5, ky0 + 0.5, cellW * 3 - 1, cellH * 3 - 1);
                ctx.fillStyle = 'rgba(255, 226, 95, 0.1)';
                ctx.fillRect(kx0 + 0.5, ky0 + 0.5, cellW * 3 - 1, cellH * 3 - 1);
            }
        }

        function tick(now) {
            if (phase === 'scan' && now - lastTime > stepMs) {
                if (scanIdx < scanOrder.length) {
                    const { y, x } = scanOrder[scanIdx];
                    const val = convAt(y, x);
                    const act = Math.abs(val) + (binary[y][x] ? 0.35 : 0.08);
                    reveal[y][x] = Math.min(1, reveal[y][x] + 0.2 + act * 0.22);
                    padded[y + 1][x + 1] = binary[y][x] ? 1 : Math.min(1, Math.abs(val));
                    scanIdx++;
                    lastTime = now;
                    if (captionEl) {
                        captionEl.textContent = `Conv2d · scan ${scanIdx}/${scanOrder.length}`;
                    }
                } else {
                    phase = 'done';
                    if (captionEl) captionEl.textContent = 'feature map · ABOUT ME';
                }
            }
            if (phase === 'done') {
                for (let y = 0; y < ROWS; y++) {
                    for (let x = 0; x < cols; x++) {
                        reveal[y][x] = Math.min(1, reveal[y][x] + 0.035);
                    }
                }
            }
            draw();
            rafId = requestAnimationFrame(tick);
        }

        function start() {
            if (rafId) cancelAnimationFrame(rafId);
            resetState();
            lastTime = performance.now();
            if (captionEl) captionEl.textContent = 'Conv2d · k=3 · s=1 · p=1';
            rafId = requestAnimationFrame(tick);
        }

        vizReplay.about = start;
        observeViz(canvas.closest('.section-viz') || canvas, start);
        resetState();
        draw();
    }

    // ── Gradient Descent — real AdamW on L(θ), experience = training steps ──
    function initGradientDescent() {
        const canvas = document.getElementById('viz-gd');
        const captionEl = document.getElementById('gd-caption');
        const card1 = document.getElementById('exp-card-1');
        const card2 = document.getElementById('exp-card-2');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const W = canvas.width;
        const H = canvas.height;

        // Global minimum θ* — "best self" after years of training
        const THETA_STAR = 0.84;
        const B1 = 0.9;
        const B2 = 0.999;
        const EPS = 1e-8;
        const WD = 0.012;
        const ETA_MAX = 0.088;
        const ETA_MIN = 0.004;
        const T_MAX = 110;
        const WARMUP = 8;

        // L(θ): convex bowl + local minima (early career traps)
        function loss(theta) {
            const bowl = 2.8 * Math.pow(theta - THETA_STAR, 2);
            const trap1 = 0.55 * Math.exp(-Math.pow((theta - 0.30) / 0.072, 2));
            const trap2 = 0.38 * Math.exp(-Math.pow((theta - 0.52) / 0.055, 2));
            const ripple = 0.05 * Math.sin(theta * 28) * Math.exp(-Math.pow(theta - 0.45, 2) * 8);
            return bowl + trap1 + trap2 + ripple + 0.18;
        }

        function grad(theta) {
            const h = 1e-4;
            return (loss(theta + h) - loss(theta - h)) / (2 * h);
        }

        // Cosine annealing with linear warmup: η_t = η_min + ½(η_max − η_min)(1 + cos(πt/T))
        function learningRate(step) {
            const warm = Math.min(step / WARMUP, 1);
            const progress = Math.min(step / T_MAX, 1);
            const cosine = ETA_MIN + 0.5 * (ETA_MAX - ETA_MIN) * (1 + Math.cos(Math.PI * progress));
            return cosine * warm;
        }

        // AdamW: m̂/(√v̂+ε) step + decoupled weight decay
        function adamwStep(state) {
            const { theta, m, v, step } = state;
            const t = step + 1;
            const eta = learningRate(t);
            const g = grad(theta);
            const mNew = B1 * m + (1 - B1) * g;
            const vNew = B2 * v + (1 - B2) * g * g;
            const mHat = mNew / (1 - Math.pow(B1, t));
            const vHat = vNew / (1 - Math.pow(B2, t));
            const thetaNew = theta - eta * (mHat / (Math.sqrt(vHat) + EPS) + WD * theta);
            return {
                theta: Math.max(0.06, Math.min(0.94, thetaNew)),
                m: mNew,
                v: vNew,
                step: t,
                eta,
                g,
                loss: loss(thetaNew)
            };
        }

        const bounds = (() => {
            let minL = Infinity, maxL = -Infinity;
            for (let i = 0; i <= 140; i++) {
                const l = loss(i / 140);
                minL = Math.min(minL, l);
                maxL = Math.max(maxL, l);
            }
            return { padL: 8, padR: 8, padT: 8, padB: 14, minL: minL - 0.06, maxL: maxL + 0.1 };
        })();

        function plot(theta, lv) {
            const cw = W - bounds.padL - bounds.padR;
            const ch = H - bounds.padT - bounds.padB;
            return {
                cx: bounds.padL + theta * cw,
                cy: bounds.padT + ((lv - bounds.minL) / (bounds.maxL - bounds.minL)) * ch,
                cw, ch
            };
        }

        const L0 = loss(0.08);
        const LStar = loss(THETA_STAR);
        const stepMs = 95;

        let sim, display, trail, checkpoint1, checkpoint2, converged, started, lastStepTime, rafId;

        function lossColor(l) {
            const t = Math.max(0, Math.min(1, (L0 - l) / (L0 - LStar)));
            const r = Math.round(233 + (208 - 233) * t);
            const g = Math.round(69 + (255 - 69) * t);
            const b = Math.round(96 + (0 - 96) * t);
            return { rgb: `rgb(${r},${g},${b})`, glow: `rgba(${r},${g},${b},0.22)` };
        }

        function draw() {
            ctx.clearRect(0, 0, W, H);

            // L(θ) landscape
            ctx.beginPath();
            for (let i = 0; i <= 160; i++) {
                const p = plot(i / 160, loss(i / 160));
                if (i === 0) ctx.moveTo(p.cx, p.cy);
                else ctx.lineTo(p.cx, p.cy);
            }
            const curveGrad = ctx.createLinearGradient(bounds.padL, 0, W - bounds.padR, 0);
            curveGrad.addColorStop(0, 'rgba(233,69,96,0.85)');
            curveGrad.addColorStop(0.55, 'rgba(255,226,95,0.85)');
            curveGrad.addColorStop(1, 'rgba(65,199,223,0.85)');
            ctx.strokeStyle = curveGrad;
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // θ* — global optimum
            const star = plot(THETA_STAR, LStar);
            ctx.beginPath();
            ctx.arc(star.cx, star.cy, 2.8, 0, Math.PI * 2);
            ctx.fillStyle = '#d0ff00';
            ctx.fill();

            // Experience checkpoints (weights saved after each role)
            [checkpoint1, checkpoint2].forEach((ck, i) => {
                if (!ck) return;
                const p = plot(ck.theta, ck.loss);
                ctx.beginPath();
                ctx.moveTo(p.cx, bounds.padT);
                ctx.lineTo(p.cx, H - bounds.padB);
                ctx.strokeStyle = i === 0 ? 'rgba(255,226,95,0.22)' : 'rgba(65,199,223,0.22)';
                ctx.lineWidth = 1;
                ctx.setLineDash([2, 3]);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.beginPath();
                ctx.arc(p.cx, p.cy, 2, 0, Math.PI * 2);
                ctx.fillStyle = i === 0 ? '#ffe25f' : '#41c7df';
                ctx.fill();
            });

            // Optimization path (actual AdamW steps)
            if (trail.length > 1) {
                ctx.beginPath();
                trail.forEach((pt, i) => {
                    const p = plot(pt.theta, pt.loss);
                    if (i === 0) ctx.moveTo(p.cx, p.cy);
                    else ctx.lineTo(p.cx, p.cy);
                });
                ctx.strokeStyle = 'rgba(65, 199, 223, 0.4)';
                ctx.lineWidth = 1.1;
                ctx.stroke();
            }

            const pt = plot(display.theta, display.loss);
            const g = grad(display.theta);

            // −∇L arrow (steepest descent direction)
            const { cw, ch } = plot(0, 0);
            const slopePx = -g * (ch / (bounds.maxL - bounds.minL)) * 0.018;
            ctx.beginPath();
            ctx.moveTo(pt.cx, pt.cy);
            ctx.lineTo(pt.cx + 14, pt.cy + Math.max(-22, Math.min(22, slopePx * 14)));
            ctx.strokeStyle = 'rgba(255, 226, 95, 0.5)';
            ctx.lineWidth = 1;
            ctx.stroke();

            const colors = converged
                ? { rgb: '#d0ff00', glow: 'rgba(208,255,0,0.2)' }
                : lossColor(display.loss);
            ctx.beginPath();
            ctx.arc(pt.cx, pt.cy, 5.5, 0, Math.PI * 2);
            ctx.fillStyle = colors.glow;
            ctx.fill();
            ctx.beginPath();
            ctx.arc(pt.cx, pt.cy, converged ? 3.5 : 3, 0, Math.PI * 2);
            ctx.fillStyle = colors.rgb;
            ctx.shadowColor = colors.rgb;
            ctx.shadowBlur = converged ? 12 : 8;
            ctx.fill();
            ctx.shadowBlur = 0;
        }

        function updateCaption() {
            if (!captionEl) return;
            if (converged) {
                captionEl.textContent = `θ* · L=${display.loss.toFixed(3)} · converged`;
                captionEl.style.color = 'var(--lime)';
            } else {
                captionEl.textContent = `step ${sim.step} · η=${sim.eta.toFixed(4)} · L=${display.loss.toFixed(3)}`;
                captionEl.style.color = '';
            }
        }

        function resetGd() {
            sim = { theta: 0.08, m: 0, v: 0, step: 0, eta: 0, g: grad(0.08), loss: L0 };
            display = { theta: 0.08, loss: L0 };
            trail = [{ theta: 0.08, loss: L0 }];
            checkpoint1 = null;
            checkpoint2 = null;
            converged = false;
            lastStepTime = 0;
            card1?.classList.remove('exp-active');
            card2?.classList.remove('exp-active');
            if (captionEl) captionEl.style.color = '';
        }

        function startGd() {
            if (rafId) cancelAnimationFrame(rafId);
            resetGd();
            started = true;
            lastStepTime = performance.now();
            rafId = requestAnimationFrame(tick);
        }

        function tick(now) {
            if (!started) {
                rafId = requestAnimationFrame(tick);
                return;
            }

            display.theta += (sim.theta - display.theta) * 0.12;
            display.loss += (sim.loss - display.loss) * 0.12;

            if (!converged && now - lastStepTime > stepMs) {
                sim = adamwStep(sim);
                trail.push({ theta: sim.theta, loss: sim.loss });
                lastStepTime = now;

                if (!checkpoint1 && sim.step >= 22 && sim.loss < L0 * 0.75) {
                    checkpoint1 = { theta: sim.theta, loss: sim.loss };
                    card1?.classList.add('exp-active');
                }
                if (!checkpoint2 && sim.step >= 38 && sim.loss < L0 * 0.42) {
                    checkpoint2 = { theta: sim.theta, loss: sim.loss };
                    card2?.classList.add('exp-active');
                }

                const atStar = Math.abs(sim.theta - THETA_STAR) < 0.018;
                const flat = Math.abs(grad(sim.theta)) < 0.1;
                if (sim.step >= 55 && atStar && flat) converged = true;
                if (sim.step >= T_MAX) converged = true;
            }

            if (converged) {
                display.theta += (THETA_STAR - display.theta) * 0.06;
                display.loss += (LStar - display.loss) * 0.06;
            }

            updateCaption();
            draw();
            rafId = requestAnimationFrame(tick);
        }

        resetGd();
        draw();
        updateCaption();
        started = false;

        vizReplay.gd = startGd;
        observeViz(canvas.closest('.section-viz') || canvas, startGd);
    }


    // ── Section-specific ML visuals ──
    function initSectionVisuals() {
        initAboutCNN();
        initGradientDescent();
        if (window.MLAnimations) {
            MLAnimations.initSection(vizReplay);
        }
        bindReplayButtons();
    }

    // ── Keyword markup {text} #hex ──
    function parseColorMarkup() {
        document.querySelectorAll('li:not(.main-nav li), p, .hero-intro').forEach(el => {
            el.innerHTML = el.innerHTML.replace(
                /\{([^}]+)\}\s*#([0-9A-Fa-f]{6})/g,
                '<span class="kw" style="color:#$2;font-weight:600;">$1</span>'
            );
        });
    }

    // ── Boot ──
    document.addEventListener('DOMContentLoaded', () => {
        initHeroNetwork();
        initCursorGlow();
        initHeader();
        initScrollReveal();
        initExperienceTimeline();
        initProjectFilters();
        initSkills();
        parseColorMarkup();
        initNavigation();
        initSectionVisuals();
    });
})();
