document.addEventListener('DOMContentLoaded', function() {
    const progressBar = document.querySelector('.progress-bar');
    const experienceItems = document.querySelectorAll('.experience-item');
    
    const observerOptions = {
        threshold: 0.3,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
                
                const visibleItems = document.querySelectorAll('.experience-item.animate-in');
                const progressHeight = (visibleItems.length / experienceItems.length) * 100;
                progressBar.style.height = `${progressHeight}%`;
            }
        });
    }, observerOptions);
    
    experienceItems.forEach(item => {
        observer.observe(item);
    });
    
    const skillChips = document.querySelectorAll('.skill-chip');
    skillChips.forEach(chip => {
        chip.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });
        
        chip.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
});


// Smooth scrolling for navigation links
document.querySelectorAll('nav a').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        
        document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
        this.classList.add('active');

        const targetId = this.getAttribute('href');
        const targetSection = document.querySelector(targetId);
        window.scrollTo({
            top: targetSection.offsetTop - 80,
            behavior: 'smooth'
        });
    });
});

// Skills data
const skillsData = {
    languages: [
        { name: "Python", icon: "fab fa-python" },
        { name: "HTML", icon: "fab fa-html5" },
        { name: "CSS", icon: "fab fa-css3-alt" },
        { name: "JavaScript", icon: "fab fa-js" },
        { name: "C++", icon: "fas fa-code" },
        { name: "Node.js", icon: "fab fa-node-js" },
        { name: "SQL", icon: "fas fa-database" },
        { name: "TypeScript", icon: "fas fa-file-code" },
        { name: "React", icon: "fas fa-react"}
    ],
    expertise: [
        { name: "Deep Learning", icon: "fas fa-brain" },
        { name: "Machine Learning", icon: "fas fa-robot" },
        { name: "NLP", icon: "fas fa-language" },
        { name: "Diffusion Models", icon: "fas fa-cloud" },
        { name: "Generative AI", icon: "fas fa-magic" },
        { name: "Reinforcement Learning", icon: "fas fa-sync-alt" },
        { name: "Computer Vision", icon: "fas fa-eye" },
        { name: "Prompt Engineering", icon: "fas fa-keyboard" }
    ],
    platforms: [
        { name: "OpenAI Playground", icon: "fas fa-play-circle" },
        { name: "Microsoft Azure", icon: "fab fa-microsoft" },
        { name: "AWS", icon: "fab fa-aws" },
        { name: "Google Cloud", icon: "fab fa-google" },
        { name: "Hugging Face", icon: "fas fa-face-smile" },
        { name: "Vertex AI", icon: "fas fa-cube" },
        { name: "Kaggle", icon: "fab fa-kaggle" },
        { name: "Docker", icon: "fab fa-docker" }
    ],
    libraries: [
        { name: "PyTorch", icon: "fas fa-fire" },
        { name: "TensorFlow", icon: "fas fa-project-diagram" },
        { name: "Keras", icon: "fas fa-layer-group" },
        { name: "Scikit-learn", icon: "fas fa-chart-line" },
        { name: "NumPy", icon: "fas fa-calculator" },
        { name: "Pandas", icon: "fas fa-table" },
        { name: "Transformers", icon: "fas fa-exchange-alt" },
        { name: "OpenCV", icon: "fas fa-camera" },
        { name: "Matplotlib", icon: "fas fa-chart-bar" }
    ]
};

// DOM elements
const categories = document.querySelectorAll('.skill-category');
const skillsDisplay = document.getElementById('skillsDisplay');

// Function to display skills with typing animation
function displaySkills(category) {
    skillsDisplay.innerHTML = '';
    
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    skillsDisplay.appendChild(typingIndicator);
    
    const skills = skillsData[category];
    
    setTimeout(() => {
        skillsDisplay.innerHTML = '';
        
        skills.forEach(skill => {
            const skillItem = document.createElement('div');
            skillItem.className = 'skill-item';
            skillItem.innerHTML = `
                <div class="skill-icon">
                    <i class="${skill.icon}"></i>
                </div>
                <div class="skill-name">${skill.name}</div>
            `;
            skillsDisplay.appendChild(skillItem);
        });
    }, 1500);
}

function setActiveCategory(category) {
    categories.forEach(cat => cat.classList.remove('active'));
    category.classList.add('active');
    displaySkills(category.dataset.category);
}

categories.forEach(category => {
    category.addEventListener('click', () => {
        setActiveCategory(category);
    });
});

document.addEventListener('DOMContentLoaded', () => {
    displaySkills('languages');
});

window.addEventListener('scroll', function() {
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('nav a');
    
    let current = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        
        if (pageYOffset >= (sectionTop - 150)) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// Initialize with about section active
document.querySelector('nav a[href="#about"]').classList.add('active');

document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("li").forEach(li => {
      li.innerHTML = li.innerHTML.replace(/\{([^}]+)\}\s*#([0-9A-Fa-f]{6})/g, 
        '<span style="color:#$2; font-weight:500;">$1</span>');
    });
});