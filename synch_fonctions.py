# -*- coding: utf-8 -*-
# Nom du fichier: synch_fonctions.py
import numpy as np

#####################################################################
#
#  Génération de N symboles MDP-4 aléatoires 
#
#  entrées:
#  - Es: énergie moyenne par symbole
#  - N: nombre symboles
#
#  sorties:
#  - data[N]: vecteur contenant les symboles
#
####################################################################
def gendata(N,Es):
    C=[np.sqrt(Es)*np.exp(1j*np.pi/4),np.sqrt(Es)*np.exp(1j*3*np.pi/4),np.sqrt(Es)*np.exp(1j*5*np.pi/4),np.sqrt(Es)*np.exp(1j*7*np.pi/4)]
    data=np.random.choice(C,N)
    return data

#####################################################################
#
#  Génération d'un échantillon de la fonction triangle de base 2*T
#
#  entrées:
#  - t: instant d'échantillonnage
#  - T: durée symbole
#
#  sorties:
#  - g: échantillon
#
####################################################################
def triangle(t,T):
   if np.abs(t)<T:
      g=1-np.abs(t)/T
   else:
      g=0
   
   return g

#####################################################################
#
#  Génération d'un échantillon de sortie du canal à l'instant t
#
#  entrées:
#  - t: instant d'échantillonnage
#  - T: durée symbole
#  - data[N]: symboles complexes émis
#  - N0: densité spectrale monolatérale du bruit
#  - phi: déphasage introduit par le canal
#  - fD: décalage fréquentiel introduit par le canal
#  - espilon: retard fractionnaire introduit par le canal
#
#  sorties:
#  - yt: échantillon de sortie du canal à l'instant t
#
####################################################################
def RX(t,T,data,N0,phi,fD,epsilon):
   # initialiser yt
   yt=0.0
   # ajouter la contribution des N symboles
   for k in range(data.size):
      yt+=data[k]*triangle(t-k*T-epsilon*T,T)
   yt*=np.exp(1j*(phi+2*np.pi*fD*t))
   # ajouter l'effet du bruit sur le canal
   """ code python manquant """
   yt+=np.random.normal(0,np.sqrt(N0/2))+1j*np.random.normal(0,np.sqrt(N0/2))
   return yt

#####################################################################
#
#  Décision optimale au sens du maximum de vraisemblance
#  pour la constellation MDP-4
#
#  entrées:
#  - Es: énergie moyenne par symbole
#  - r: échantillon bruité
#
#  sorties:
#  - res: décision optimale
#
####################################################################
def decision(r,Es):
   """ code python manquant """
   res=np.sqrt(Es/2.0)*(np.sign(np.real(r))+1j*np.sign(np.imag(r)))
   return res


#####################################################################
#
#  Boucle à verouillage de phase
#
#  entrées:
#  - y[N]: échantillons bruités en sortie du canal
#  - gamma: gain de boucle
#  - phi_chap_0: estimée de phase initiale
#  - Es: énergie moyenne par symbole
#
#  sorties:
#  - phi_chap[N]:estimée de phase correspondant à chaque échantillon bruité
#  - a_chap[N]: estimée du symbole correspondant à chaque échantillon bruité
#
####################################################################
def PhaseEstimation(y,gamma,phi_chap_0,Es):
   # initialization des sorties
   phi_chap=np.zeros(y.size)
   a_chap=np.zeros(y.size,dtype='complex')
   
   for k in range(y.size):
      # initialisation
      if k==0:
         # initialization de la boucle à verouillage de phase
         phi_chap[k]=phi_chap_0
         # variable de décision
         zz=y[k]*np.exp(-1j*phi_chap_0)
         # decision optimale
         #m=np.trunc(-1/2-(2/np.pi)*np.angle(zz))
         #a_chap[k]=np.sqrt(Es)*np.exp(1j*m*np.pi/2)
         a_chap[k]=decision(zz,Es)
         """ code python manquant """
      # pour les instants suivants
      else:
         zz=y[k]*np.exp(-1j*phi_chap[k-1])
         a_chap[k]=decision(zz,Es)
         uk=np.imag(np.conj(a_chap[k])*zz)
         phi_chap[k]=phi_chap[k-1]+gamma*uk
         #m=np.trunc(-1/2-(2/np.pi)*np.angle(zz))
         #a_chap[k]=np.sqrt(Es)*np.exp(1j*m*np.pi/2)
         """ code python manquant """

   return phi_chap,a_chap