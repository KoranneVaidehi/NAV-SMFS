import { useEffect, useRef, useState } from 'react';

export default function About() {
  const [isVisible, setIsVisible] = useState(false);
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) setIsVisible(true);
      },
      { threshold: 0.3 }
    );
    if (sectionRef.current) observer.observe(sectionRef.current);
    return () => observer.disconnect();
  }, []);

  return (
    <section ref={sectionRef} id="about" className="section-padding" style={{ backgroundColor: 'var(--color-bg-secondary)' }}>
      <div className="container">
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
          gap: '6rem',
          alignItems: 'center'
        }}>
          
          <div style={{
            opacity: isVisible ? 1 : 0,
            transform: isVisible ? 'translateY(0)' : 'translateY(30px)',
            transition: 'var(--transition-slow)'
          }}>
            <h2 className="text-display-large" style={{ marginBottom: '2rem' }}>
              DIGITAL MEDIA<br />
              <span style={{ color: 'var(--color-text-secondary)' }}>SHOULD BE</span><br />
              VERIFIABLE.
            </h2>
            <p className="text-body-large" style={{ marginBottom: '2rem' }}>
              Advances in generative AI have made synthetic and manipulated imagery increasingly difficult to recognize by visual inspection alone. NAVSMFS approaches image authenticity as a forensic problem—examining visual evidence to determine whether an image is likely authentic or manipulated.
            </p>
            <p style={{ 
              fontSize: '1.25rem', 
              fontWeight: 500, 
              color: 'var(--color-text-primary)',
              borderLeft: '2px solid var(--color-accent)',
              paddingLeft: '1.5rem',
              lineHeight: 1.6
            }}>
              The goal is not to replace human judgment.<br />
              It is to strengthen it with evidence.
            </p>
          </div>

          <div style={{
            position: 'relative',
            aspectRatio: '3/4',
            overflow: 'hidden',
            backgroundColor: 'var(--color-bg-primary)',
            opacity: isVisible ? 1 : 0,
            transform: isVisible ? 'translateY(0)' : 'translateY(40px)',
            transition: 'all 1s cubic-bezier(0.16, 1, 0.3, 1) 0.2s',
            boxShadow: '0 20px 40px rgba(0,0,0,0.4)'
          }}>
            <div style={{
              position: 'absolute', inset: 0,
              backgroundImage: 'url(/hero-portrait.jpg)',
              backgroundSize: 'cover',
              backgroundPosition: 'center',
              filter: 'grayscale(100%) contrast(1.2)'
            }} />
            
            <div className="forensic-scan-line" />
            
            {/* Visual Overlays */}
            <div style={{
              position: 'absolute', inset: 0,
              background: 'linear-gradient(to bottom, transparent, rgba(8,8,8,0.8))'
            }} />
            
            <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }} viewBox="0 0 100 100" preserveAspectRatio="none">
              <rect x="25" y="15" width="50" height="60" fill="none" stroke="var(--color-accent)" strokeWidth="0.5" opacity="0.6" strokeDasharray="2,2" />
              <circle cx="40" cy="35" r="1.5" fill="var(--color-accent)" opacity="0.8" />
              <circle cx="60" cy="35" r="1.5" fill="var(--color-accent)" opacity="0.8" />
              <circle cx="50" cy="55" r="1.5" fill="var(--color-accent)" opacity="0.8" />
              <path d="M 40,65 Q 50,70 60,65" fill="none" stroke="var(--color-accent)" strokeWidth="0.5" opacity="0.8" />
            </svg>
            
            <div style={{ position: 'absolute', bottom: '2rem', left: '2rem' }}>
              <p className="text-label-small" style={{ color: 'var(--color-accent)' }}>FACIAL LANDMARK DETECTION</p>
              <p style={{ fontFamily: 'monospace', fontSize: '0.75rem', marginTop: '0.5rem', opacity: 0.6 }}>MODEL: MTCNN_V2 // ACCURACY: HIGH</p>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
}
