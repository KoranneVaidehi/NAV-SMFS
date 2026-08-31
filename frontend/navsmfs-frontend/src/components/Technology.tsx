

export default function Technology() {
  return (
    <section className="section-padding" style={{ backgroundColor: 'var(--color-bg-secondary)' }}>
      <div className="container">
        
        <div style={{ textAlign: 'center', marginBottom: '6rem' }}>
          <p className="text-label-small" style={{ marginBottom: '1rem' }}>CORE TECHNOLOGY STACK</p>
          <h2 className="text-display-large">FORENSIC ENGINE</h2>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
          
          <div style={{ 
            padding: '4rem', 
            backgroundColor: 'var(--color-bg-primary)',
            borderTop: '2px solid var(--color-bg-tertiary)' 
          }}>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '2.5rem', marginBottom: '1rem' }}>MTCNN</h3>
            <p className="text-body-large">Facial detection and localization.</p>
            <div style={{ marginTop: '2rem', height: '1px', width: '40px', backgroundColor: 'var(--color-accent)' }} />
          </div>

          <div style={{ 
            padding: '4rem', 
            backgroundColor: 'var(--color-bg-primary)',
            borderTop: '2px solid var(--color-accent)' 
          }}>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '2.5rem', marginBottom: '1rem' }}>OpenCV</h3>
            <p className="text-body-large">Computer-vision based image processing and forensic analysis.</p>
            <div style={{ marginTop: '2rem', height: '1px', width: '40px', backgroundColor: 'var(--color-text-secondary)' }} />
          </div>

        </div>
        
      </div>
    </section>
  );
}
