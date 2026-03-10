import Navbar from './components/Navbar'
import Hero from './components/Hero'
import Features from './components/Features'
import Philosophy from './components/Philosophy'
import Protocol from './components/Protocol'
import Simulator from './components/Simulator'
import CTA from './components/CTA'
import Footer from './components/Footer'

export default function App() {
  return (
    <div className="font-grotesk bg-paper text-ink">
      <Navbar />
      <Hero />
      <Features />
      <Philosophy />
      <Protocol />
      <Simulator />
      <CTA />
      <Footer />
    </div>
  )
}
