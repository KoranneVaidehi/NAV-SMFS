
import Navigation from './components/Navigation';
import Hero from './components/Hero';
import About from './components/About';
import HowItWorks from './components/HowItWorks';
import Technology from './components/Technology';
import Scanner from './components/Scanner';

function App() {
  return (
    <>
      <Navigation />
      <main>
        <Hero />
        <About />
        <HowItWorks />
        <Technology />
        <Scanner />
      </main>
      <footer style={{ padding: '4rem 5%', borderTop: '1px solid var(--color-bg-tertiary)', marginTop: '8rem' }}>
        <div className="container" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', alignItems: 'center', textAlign: 'center' }}>
          <h4 style={{ color: 'var(--color-text-primary)' }}>NAVSMFS</h4>
          <p className="text-label-small">Neural Authentication Verification and Synthetic Media Forensics System</p>
          <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.875rem' }}>AI-assisted image authentication and synthetic-media analysis.</p>
          <div style={{ display: 'flex', gap: '2rem', marginTop: '2rem', fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>
            <a href="#home">Home</a>
            <a href="#about">About Us</a>
            <a href="#how-it-works">How It Works</a>
            <a href="#scan">Scan & Verify</a>
          </div>
          <p style={{ marginTop: '2rem', color: 'var(--color-text-secondary)', fontSize: '0.75rem' }}>© NAVSMFS</p>
        </div>
      </footer>
    </>
  );
}

export default App;
