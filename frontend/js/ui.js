/**
 * MediFusion AI — UI Utilities, Toasts, Counter Animations, Validation Overlays, and Canvas Visuals
 * Premium Healthcare AI Platform
 */

const MediFusionUI = (function () {
    "use strict";

    /**
     * Enhanced Toast Notifications with progress bar
     */
    function showToast(type, message, durationMs = 4000) {
        let container = document.getElementById("toast-container");
        if (!container) {
            container = document.createElement("div");
            container.id = "toast-container";
            document.body.appendChild(container);
        }

        const toast = document.createElement("div");
        toast.className = `toast ${type}`;

        const icons = {
            success: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6L9 17l-5-5"/></svg>`,
            error: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`,
            info: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`,
            warning: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
        };

        toast.innerHTML = `
            <span class="toast-icon-wrap" style="display:flex;align-items:center;">${icons[type] || icons.info}</span>
            <div style="flex:1;">${message}</div>
            <div class="toast-progress" style="animation-duration:${durationMs}ms;"></div>
        `;

        container.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = "toast-slide-out 0.3s ease forwards";
            setTimeout(() => toast.remove(), 300);
        }, durationMs);
    }

    /**
     * Show Validation Error Overlay on a container (e.g., dropzone)
     * Returns the overlay element for external management
     */
    function showValidationError(containerEl, title, message, autoDismissMs = 6000) {
        // Remove any existing validation overlay
        const existing = containerEl.querySelector(".validation-error-overlay");
        if (existing) existing.remove();

        const overlay = document.createElement("div");
        overlay.className = "validation-error-overlay";
        overlay.innerHTML = `
            <div class="val-error-icon">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
            </div>
            <div class="val-error-title">${title}</div>
            <div class="val-error-msg">${message}</div>
            <button class="val-dismiss-btn">Dismiss</button>
        `;

        // Dismiss button handler
        const dismissBtn = overlay.querySelector(".val-dismiss-btn");
        dismissBtn.addEventListener("click", () => {
            overlay.style.animation = "validation-fade-in 0.25s ease reverse forwards";
            setTimeout(() => overlay.remove(), 250);
        });

        containerEl.appendChild(overlay);

        // Auto-dismiss after timeout
        if (autoDismissMs > 0) {
            setTimeout(() => {
                if (overlay.parentElement) {
                    overlay.style.animation = "validation-fade-in 0.25s ease reverse forwards";
                    setTimeout(() => overlay.remove(), 250);
                }
            }, autoDismissMs);
        }

        return overlay;
    }

    /**
     * Smooth Number Count-Up Animation
     */
    function animateNumber(element, start, end, duration = 1000, suffix = "%", decimals = 1) {
        if (!element) return;
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out quart
            const easeProgress = 1 - Math.pow(1 - progress, 4);
            const current = start + (end - start) * easeProgress;

            element.textContent = `${current.toFixed(decimals)}${suffix}`;

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                element.textContent = `${end.toFixed(decimals)}${suffix}`;
            }
        }

        requestAnimationFrame(update);
    }

    /**
     * Initialize Scroll Reveal Animations (IntersectionObserver)
     */
    function initScrollReveal() {
        const revealElements = document.querySelectorAll(".scroll-reveal");
        if (revealElements.length === 0) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("revealed");
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.15,
            rootMargin: "0px 0px -60px 0px",
        });

        revealElements.forEach((el) => observer.observe(el));
    }

    /**
     * Initialize Counter Animations (triggered by IntersectionObserver)
     */
    function initCounterAnimations() {
        const counters = document.querySelectorAll(".counter-animate");
        if (counters.length === 0) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    const el = entry.target;
                    const target = parseInt(el.dataset.target, 10);
                    const suffix = el.dataset.suffix || "";
                    const duration = 1800;

                    if (isNaN(target)) return;

                    const startTime = performance.now();

                    function animate(currentTime) {
                        const elapsed = currentTime - startTime;
                        const progress = Math.min(elapsed / duration, 1);
                        const easeProgress = 1 - Math.pow(1 - progress, 4);
                        const current = Math.round(easeProgress * target);

                        el.textContent = current.toLocaleString() + suffix;

                        if (progress < 1) {
                            requestAnimationFrame(animate);
                        } else {
                            el.textContent = target.toLocaleString() + suffix;
                        }
                    }

                    requestAnimationFrame(animate);
                    observer.unobserve(el);
                }
            });
        }, {
            threshold: 0.3,
        });

        counters.forEach((el) => observer.observe(el));
    }

    /**
     * Animated Neural Particle Canvas
     */
    function initNeuralCanvas(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        const ctx = canvas.getContext("2d");
        let width = (canvas.width = canvas.parentElement.offsetWidth);
        let height = (canvas.height = canvas.parentElement.offsetHeight);

        const particles = [];
        const particleCount = Math.min(Math.floor(width / 22), 55);

        for (let i = 0; i < particleCount; i++) {
            particles.push({
                x: Math.random() * width,
                y: Math.random() * height,
                vx: (Math.random() - 0.5) * 0.4,
                vy: (Math.random() - 0.5) * 0.4,
                radius: Math.random() * 1.8 + 1,
            });
        }

        window.addEventListener("resize", () => {
            if (!canvas.parentElement) return;
            width = canvas.width = canvas.parentElement.offsetWidth;
            height = canvas.height = canvas.parentElement.offsetHeight;
        });

        function render() {
            ctx.clearRect(0, 0, width, height);

            for (let i = 0; i < particles.length; i++) {
                const p = particles[i];
                p.x += p.vx;
                p.y += p.vy;

                if (p.x < 0 || p.x > width) p.vx *= -1;
                if (p.y < 0 || p.y > height) p.vy *= -1;

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fillStyle = "rgba(45, 212, 191, 0.45)";
                ctx.fill();

                for (let j = i + 1; j < particles.length; j++) {
                    const p2 = particles[j];
                    const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
                    if (dist < 130) {
                        ctx.beginPath();
                        ctx.moveTo(p.x, p.y);
                        ctx.lineTo(p2.x, p2.y);
                        const alpha = (1 - dist / 130) * 0.18;
                        ctx.strokeStyle = `rgba(45, 212, 191, ${alpha})`;
                        ctx.lineWidth = 0.75;
                        ctx.stroke();
                    }
                }
            }

            requestAnimationFrame(render);
        }

        render();
    }

    // Auto-initialize on DOM ready
    document.addEventListener("DOMContentLoaded", () => {
        initScrollReveal();
        initCounterAnimations();
    });

    return {
        showToast,
        showValidationError,
        animateNumber,
        initNeuralCanvas,
        initScrollReveal,
        initCounterAnimations,
    };
})();

window.MediFusionUI = MediFusionUI;
