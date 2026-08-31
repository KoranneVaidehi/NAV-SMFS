import { useEffect, useState } from 'react';
import { ScanSearch } from 'lucide-react';

export default function Navigation() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 100,
      padding: '1.5rem 5%', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      transition: 'var(--transition-normal)',
      background: scrolled ? 'rgba(8, 8, 8, 0.9)' : 'transparent',
      backdropFilter: scrolled ? 'blur(12px)' : 'none',
      borderBottom: scrolled ? '1px solid var(--color-bg-tertiary)' : '1px solid transparent'
    }}>
      <div style={{ fontFamily: 'var(--font-display)', fontWeight: 600, fontSize: '1.25rem', letterSpacing: '0.05em' }}>NAVSMFS</div>
      
      <div style={{ display: 'flex', gap: '3rem', fontSize: '0.875rem', fontWeight: 500, color: 'var(--color-text-secondary)' }} className="desktop-nav">
        <a href="#home" style={{ color: 'var(--color-text-primary)' }}>Home</a>
        <a href="#about">About Us</a>
        <a href="#how-it-works">How It Works</a>
        <a href="#scan">Scan & Verify</a>
      </div>

      <a href="#scan" className="btn-primary" style={{ padding: '0.75rem 1.5rem', fontSize: '0.75rem' }}>
        SCAN IMAGE <ScanSearch size={16} />
      </a>
    </nav>
  );
}
