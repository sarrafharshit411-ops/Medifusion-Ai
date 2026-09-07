/**
 * MediFusion AI — Global Navigation & Telemetry Module
 */

(function () {
    "use strict";

    document.addEventListener("DOMContentLoaded", () => {
        initNavigation();
    });

    function initNavigation() {
        // Highlight active link based on current pathname
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll(".nav-item");

        navLinks.forEach((link) => {
            const href = link.getAttribute("href");
            if (
                (currentPath === "/" && href === "/") ||
                (href !== "/" && currentPath.includes(href.replace(".html", "")))
            ) {
                link.classList.add("active");
            } else {
                link.classList.remove("active");
            }
        });

        // Mobile drawer toggle
        const toggleBtn = document.getElementById("mobileToggle");
        const drawer = document.getElementById("mobileDrawer");
        if (toggleBtn && drawer) {
            toggleBtn.addEventListener("click", () => {
                drawer.classList.toggle("open");
            });
        }
    }
})();
