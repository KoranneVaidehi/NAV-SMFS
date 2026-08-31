

export default function HowItWorks() {
  const steps = [
    {
      num: "01",
      title: "UPLOAD",
      desc: "Select an image for verification. Supported formats: JPG, JPEG, PNG, WEBP.",
      img: "/hero-portrait.jpg"
    },
    {
      num: "02",
      title: "ANALYZE",
      desc: "NAVSMFS processes the image through its current computer-vision pipeline. MTCNN is used for facial detection and localization. OpenCV is used for image processing and forensic analysis.",
      img: "/hero-portrait.jpg",
      isAnalyze: true
    },
    {
      num: "03",
      title: "VERIFY",
      desc: "The system produces an authentication result with a confidence-oriented interpretation and visual evidence.",
      img: "/hero-portrait.jpg"
    }
  ];

  return (
    <section id="how-it-works" className="section-padding">
      <div className="container">
        <h2 className="text-display-huge" style={{ marginBottom: '6rem', textAlign: 'center' }}>
          FROM IMAGE<br />
          <span style={{ color: 'var(--color-text-secondary)' }}>TO EVIDENCE.</span>
        </h2>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8rem' }}>
          {steps.map((step, idx) => (
            <div key={step.num} style={{
              display: 'flex',
              flexDirection: idx % 2 !== 0 ? 'row-reverse' : 'row',
              flexWrap: 'wrap',
              gap: '4rem',
              alignItems: 'center'
            }}>
              
              <div style={{ flex: '1 1 400px' }}>
                <p className="text-label-small" style={{ color: 'var(--color-accent)', marginBottom: '1rem' }}>STEP {step.num}</p>
                <h3 className="text-display-medium" style={{ marginBottom: '1.5rem' }}>{step.title}</h3>
                <p className="text-body-large">{step.desc}</p>
              </div>
              
              <div style={{ flex: '1 1 400px', position: 'relative' }}>
                <div style={{
                  position: 'relative',
                  width: '100%',
                  aspectRatio: '16/9',
                  backgroundColor: 'var(--color-bg-secondary)',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    position: 'absolute', inset: 0,
                    backgroundImage: `url(${step.img})`,
                    backgroundSize: 'cover',
                    backgroundPosition: 'center',
                    filter: step.isAnalyze ? 'sepia(30%) hue-rotate(180deg) contrast(1.1)' : 'grayscale(20%)'
                  }} />
                  
                  {step.isAnalyze && (
                    <>
                      <div className="forensic-scan-line" />
                      <div style={{ position: 'absolute', inset: 0, background: 'rgba(77,208,225,0.1)' }} />
                    </>
                  )}

                  {step.num === "03" && (
                    <div style={{
                      position: 'absolute', inset: 0,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      background: 'rgba(8,8,8,0.7)', backdropFilter: 'blur(4px)'
                    }}>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '3rem', fontFamily: 'var(--font-display)', color: 'var(--color-accent)' }}>92.8%</div>
                        <div className="text-label-small">LIKELY AUTHENTIC</div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
