// Experience Section Horizontal Scrolling
const experienceTrack = document.querySelector('.experience-track');
const leftArrow = document.querySelector('.left-arrow');
const rightArrow = document.querySelector('.right-arrow');
const experienceCards = document.querySelectorAll('.experience-card');
let currentCardIndex = 0;

// Initialize arrows
function updateArrows() {
    leftArrow.classList.toggle('hidden', currentCardIndex === 0);
    rightArrow.classList.toggle('hidden', currentCardIndex === experienceCards.length - 1);
}

// Scroll to specific card
function scrollToCard(index) {
    currentCardIndex = index;
    experienceTrack.scrollTo({
        left: experienceCards[index].offsetLeft,
        behavior: 'smooth'
    });
    updateArrows();
}

// Next card
rightArrow.addEventListener('click', () => {
    if (currentCardIndex < experienceCards.length - 1) {
        scrollToCard(currentCardIndex + 1);
    }
});

// Previous card
leftArrow.addEventListener('click', () => {
    if (currentCardIndex > 0) {
        scrollToCard(currentCardIndex - 1);
    }
});

// Initialize arrows on load
updateArrows();

// Scroll snapping detection
experienceTrack.addEventListener('scroll', () => {
    const scrollPosition = experienceTrack.scrollLeft;
    const cardWidth = experienceCards[0].offsetWidth;
    currentCardIndex = Math.round(scrollPosition / cardWidth);
    updateArrows();
});


// Smooth scrolling for navigation links
document.querySelectorAll('nav a').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        
        // Update active class
        document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
        this.classList.add('active');
        
        // Scroll to section
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
        { name: "TypeScript", icon: "fas fa-file-code" }
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
    // Clear current skills
    skillsDisplay.innerHTML = '';
    
    // Show typing indicator
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    skillsDisplay.appendChild(typingIndicator);
    
    // Get skills for the selected category
    const skills = skillsData[category];
    
    // After delay, display skills
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
    }, 1500); // Animation duration
}

// Set active category
function setActiveCategory(category) {
    categories.forEach(cat => cat.classList.remove('active'));
    category.classList.add('active');
    displaySkills(category.dataset.category);
}

// Event listeners
categories.forEach(category => {
    category.addEventListener('click', () => {
        setActiveCategory(category);
    });
});

// Initialize with languages category active
document.addEventListener('DOMContentLoaded', () => {
    displaySkills('languages');
});


// Add scroll event to highlight active section
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